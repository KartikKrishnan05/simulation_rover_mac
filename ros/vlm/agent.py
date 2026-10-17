#!/usr/bin/env python3
"""VLM-Agent: Sprachbefehl → Rover-Aktion (lokales Modell über Ollama).

  pixi run agent "go to the meeting room"
  pixi run agent "find the blue bag"
  pixi run agent "follow the person"
  pixi run agent                       # interaktiv, Befehle eintippen

Aufbau – das VLM trifft die Entscheidungen, Nav2 fährt:

  Anweisung ──VLM──► Aufgabe (goto | find | follow), Raum, Zielobjekt
  goto    Anfahrpose des Raums (semantic_map.yaml) → Nav2 → VLM prüft das Kamerabild
  find    VLM ordnet die Räume (Draufsicht der Karte + Beschreibungen) →
          Aussichtspunkte abfahren → VLM sucht im Kamerabild (Bounding-Box) →
          Rückfrage am Bildausschnitt → Tiefenbild + TF → Kartenpunkt →
          Nav2 fährt bis 1,5 m davor
  follow  wie find (Person finden) → Lidar-Tracker übernimmt (10 Hz) →
          Nav2-Ziel 1,8 m hinter der Person, laufend ersetzt → bei
          verlorener Spur sucht das VLM im Kamerabild

Während das VLM nachdenkt, steht der Rover und die Simulation ist pausiert
(--no-pause schaltet das ab). Jede Ausführung wird als Episode für ein
späteres VLA aufgezeichnet (episode_recorder.py, --no-record).

--perception oracle ersetzt das VLM durch Ground Truth aus Gazebo
(sichtbar = im Bild und nicht verdeckt). So entstehen schnell viele
Experten-Episoden mit demselben Ablauf (siehe collect.py).
"""
import argparse
import difflib
import json
import math
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GroundTruth, Navigator, OccupancyMap, SimControl,  # noqa: E402
                    load_semantic_map, make_pose, room_at, yaw_of)
from perception import (Camera, annotate, box_to_map, crop_box, oracle_box,  # noqa: E402
                        render_map_view, valid_box)
from tracker import PersonTracker  # noqa: E402
from vlm_client import DEFAULT_MODEL, VLM  # noqa: E402

TASKS = ('goto', 'find', 'follow')
# Objekte, die das Orakel kennt: Ground-Truth-Name, Höhe, Radius (m)
ORACLE_OBJECTS = {'bag': ('bag', 0.52, 0.2), 'person': ('person', 1.75, 0.25)}
APPROACH_DIST = 1.5   # m base_footprint → Objekt (Front des Rovers ~0,65 m davor)
FOLLOW_DIST = 1.8     # m hinter der Person
HOLD_DIST = 2.0       # näher: anhalten
# Gegenprobe: ab dieser Wahrscheinlichkeit für "yes" gilt ein Treffer als bestätigt.
# Das 4B-Modell ist sich bei Fehlalarmen meist sehr sicher (P(yes) < 1 %), bei
# echten Treffern aber manchmal knapp unter 50 %.
VERIFY_THRESHOLD = 0.2
RESUME_DIST = 2.6     # weiter weg: wieder losfahren

GREEN, RED, ORANGE = (60, 200, 60), (40, 40, 230), (0, 160, 255)


def oracle_key(target):
    t = (target or '').lower()
    if any(w in t for w in ('person', 'man', 'woman', 'human', 'someone', 'people')):
        return 'person'
    if any(w in t for w in ('bag', 'backpack', 'rucksack', 'tasche')):
        return 'bag'
    return None


class Agent:
    def __init__(self, perception='vlm', model=DEFAULT_MODEL, record=True, pause=True):
        rclpy.init()
        self.node = rclpy.create_node(
            'vlm_agent', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.log = self.node.get_logger()
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.node)
        self.sm = load_semantic_map()
        self.rooms = self.sm['rooms']
        self.occ = OccupancyMap()
        self.cam = Camera(self.node, self.tf)
        self.tracker = PersonTracker(self.node, self.tf, self.occ)
        self.sim = SimControl(self.node)
        self.gt = GroundTruth(self.node)
        self.nav = Navigator(self.node)
        self.bridge = CvBridge()
        self.perception = perception
        self.record = record
        self.pause_sim = pause
        self.vlm = VLM(model, on_call=self.on_vlm_call)
        self.marks = {}       # Name → (x, y, Beschriftung) für RViz und die Kartenansicht
        self.watch_hit = None  # Treffer der Suche während der Fahrt
        self.person_state = 'unknown'
        self.status = ''

        n = self.node
        self.pub_image = n.create_publisher(Image, '/vlm/image', 1)
        self.pub_map_view = n.create_publisher(Image, '/vlm/map_view', 1)
        self.pub_markers = n.create_publisher(MarkerArray, '/vlm/markers', 1)
        self.pub_subgoal = n.create_publisher(PoseStamped, '/vlm/subgoal', 10)
        self.pub_events = n.create_publisher(String, '/vlm/events', 50)
        self.pub_episode = n.create_publisher(String, '/episode/control', 10)
        self.pub_person_cmd = n.create_publisher(String, '/person/command', 10)
        self.pub_initialpose = n.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.pub_cmd = n.create_publisher(Twist, '/cmd_vel', 10)
        self.clear_costmaps = [
            n.create_client(ClearEntireCostmap, f'/{c}/clear_entirely_{c}')
            for c in ('global_costmap', 'local_costmap')]
        n.create_subscription(String, '/person/state',
                              lambda m: setattr(self, 'person_state', m.data), 10)

        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    # ================================================================ Helfer
    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def rover(self):
        t = self.tf.lookup_transform('map', 'base_footprint', Time())
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def say(self, text):
        self.status = text
        print(f'[agent] {text}', flush=True)

    def event(self, kind, **data):
        self.pub_events.publish(String(data=json.dumps({'kind': kind, **data}, default=float)))

    def on_vlm_call(self, info):
        self.event('vlm_call', **info)
        p = f' (P(yes) = {info["p_yes"]:.2f})' if 'p_yes' in info else ''
        print(f'        VLM ({info["purpose"]}, {info["seconds"]:.1f} s): {info["response"][:160]}{p}',
              flush=True)

    def publish_image(self, img):
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def publish_markers(self, person=None, goal=None):
        arr = MarkerArray()
        arr.markers.append(Marker(action=Marker.DELETEALL))
        stamp = self.node.get_clock().now().to_msg()

        def marker(mid, mtype, x, y, z, scale, rgba, text=''):
            m = Marker()
            m.header.frame_id, m.header.stamp = 'map', stamp
            m.ns, m.id, m.type = 'vlm', mid, mtype
            m.pose.position = Point(x=float(x), y=float(y), z=float(z))
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.text = text
            return m

        for i, (x, y, label) in enumerate(self.marks.values()):
            arr.markers.append(marker(10 + 2 * i, Marker.SPHERE, x, y, 0.3, (0.35, 0.35, 0.35),
                                      (0.1, 0.3, 1.0, 0.9)))
            arr.markers.append(marker(11 + 2 * i, Marker.TEXT_VIEW_FACING, x, y, 0.9,
                                      (0.0, 0.0, 0.35), (1.0, 1.0, 1.0, 1.0), label))
        if person is not None:
            arr.markers.append(marker(1, Marker.CYLINDER, person[0], person[1], 0.9, (0.45, 0.45, 1.8),
                                      (1.0, 0.5, 0.0, 0.7)))
        if goal is not None:
            arr.markers.append(marker(2, Marker.SPHERE, goal[0], goal[1], 0.1, (0.2, 0.2, 0.2),
                                      (0.0, 1.0, 0.3, 0.9)))
        try:
            x, y, _ = self.rover()
            arr.markers.append(marker(3, Marker.TEXT_VIEW_FACING, x, y, 1.6, (0.0, 0.0, 0.3),
                                      (1.0, 1.0, 0.3, 1.0), self.status[:60]))
        except TransformException:
            pass
        self.pub_markers.publish(arr)

    @contextmanager
    def thinking(self):
        """Simulation anhalten, solange das VLM rechnet (nur wenn der Rover steht)."""
        with self.sim.paused(self.pause_sim and not self.nav.active()):
            yield

    def wait_ready(self, timeout=120.0):
        self.say('warte auf Nav2, Lokalisierung und Kamera …')
        t0 = time.monotonic()
        while True:
            try:
                self.rover()
                break
            except TransformException:
                pass
            if time.monotonic() - t0 > timeout:
                raise RuntimeError('keine Lokalisierung – laufen ./run.sh und pixi run nav?')
            time.sleep(0.5)
        self.cam.snapshot(timeout=30.0)  # wirft, wenn keine Kamerabilder kommen
        if not self.nav.client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('Nav2 nicht erreichbar')
        if self.perception == 'vlm':
            err = self.vlm.check()
            if err:
                raise RuntimeError(err)
            self.say(f'lade {self.vlm.model} …')
            self.vlm.warmup()
        self.say('bereit')

    # ================================================================ Fahren
    def go(self, x, y, yaw, label='', timeout=300.0, retry=True, via=()):
        """Mit Nav2 zu einer Pose fahren (optional über via-Posen); True bei Erfolg.

        Schlägt die Fahrt fehl (meist: Rover hängt an einer Türkante und Nav2
        meldet "Start occupied"), setzt sich der Rover ein Stück frei und
        versucht es noch einmal.
        """
        stamp = self.node.get_clock().now().to_msg()
        pose = make_pose(x, y, yaw, stamp)
        self.pub_subgoal.publish(pose)
        self.event('navigate', label=label, goal=[x, y, yaw], via=[list(v) for v in via])
        self.say(f'fahre zu {label or f"({x:.1f}, {y:.1f})"}' + (' (über via)' if via else ''))
        if via:
            sent = self.nav.send_through([make_pose(*v, stamp) for v in via] + [pose])
        else:
            sent = self.nav.send(pose)
        if not sent:
            self.say('Nav2 hat das Ziel abgelehnt')
            return False
        ok = self.nav.wait(timeout, tick=lambda: self.publish_markers(goal=(x, y)))
        self.event('navigate_result', label=label, success=ok)
        if not ok and self.watch_hit is not None:
            return False  # vom Beobachter abgebrochen: Ziel gesehen
        if not ok and retry:
            self.say(f'Navigation zu {label} fehlgeschlagen – setze mich frei und versuche es erneut')
            self.unstick()
            return self.go(x, y, yaw, label, timeout, retry=False, via=via)
        if not ok:
            self.say(f'Navigation zu {label} fehlgeschlagen')
        return ok

    def unstick(self, dist=0.4, speed=0.15):
        """Ein Stück geradeaus vor oder zurück – in die Richtung mit mehr Platz."""
        x, y, yaw = self.rover()
        c, s = math.cos(yaw), math.sin(yaw)
        clear = {d: self.occ.footprint_clearance(x + d * dist * c, y + d * dist * s, yaw)
                 for d in (1, -1)}
        d = max(clear, key=clear.get)
        self.event('unstick', direction=d, clearance=clear[d])
        cmd = Twist()
        cmd.linear.x = d * speed
        t_end = self.now() + dist / speed
        while self.now() < t_end:
            self.pub_cmd.publish(cmd)
            time.sleep(0.1)
        self.pub_cmd.publish(Twist())
        for srv in self.clear_costmaps:
            if srv.wait_for_service(timeout_sec=1.0):
                srv.call_async(ClearEntireCostmap.Request())
        time.sleep(1.0)

    def go_pose(self, pose_deg, label):
        x, y, yaw = pose_deg
        return self.go(x, y, math.radians(yaw), label)

    def enter(self, room, pose_deg, label):
        """Pose in einem Raum anfahren – von außen über dessen via-Posen."""
        x, y, _ = self.rover()
        via = []
        if room_at(self.sm, x, y) != room:
            via = [(vx, vy, math.radians(vyaw)) for vx, vy, vyaw in self.rooms[room].get('via', [])]
        px, py, pyaw = pose_deg
        return self.go(px, py, math.radians(pyaw), label, via=via)

    def reset_rover(self, pose_deg=None):
        """Rover per Gazebo an eine Pose setzen und AMCL neu initialisieren.

        Die Person geht dabei an ihren Parkplatz – sonst steht sie nach einer
        Folge-Episode am Ende ihres Wegs direkt vor der Startpose.
        """
        x, y, yaw_deg = pose_deg or self.sm['start']
        yaw = math.radians(yaw_deg)
        self.nav.cancel()
        self.pub_person_cmd.publish(String(data='park'))
        # Modellursprung = Fahrzeugmitte, 0,28 m vor base_footprint
        self.sim.set_pose('rover', x + 0.28 * math.cos(yaw), y + 0.28 * math.sin(yaw), yaw, z=0.02)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose = make_pose(x, y, yaw).pose
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.03
        self.pub_initialpose.publish(msg)
        time.sleep(2.0)
        # Warten, bis AMCL die neue Pose übernommen hat – sonst hält enter() den
        # Rover noch für im alten Raum und lässt z. B. die via-Posen weg.
        t_end = time.monotonic() + 20.0
        while time.monotonic() < t_end:
            try:
                rx, ry, _ = self.rover()
                if math.hypot(rx - x, ry - y) < 0.3:
                    break
            except TransformException:
                pass
            time.sleep(1.0)
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.pub_initialpose.publish(msg)  # falls AMCL die erste verpasst hat
        else:
            self.say('AMCL hat die Startpose noch nicht übernommen')
        for srv in self.clear_costmaps:
            if srv.wait_for_service(timeout_sec=1.0):
                srv.call_async(ClearEntireCostmap.Request())
        time.sleep(1.0)

    def look(self):
        """Frisches Kamerabild, nachdem der Rover steht."""
        snap = self.cam.snapshot(after=self.now() + 0.2)
        if self.cam.wait_seconds > 3.0:
            self.log.warn(f'Kamerabild erst nach {self.cam.wait_seconds:.1f} s')
        return snap

    # ================================================================ VLM-Entscheidungen
    def room_list(self):
        return '\n'.join(f'- {n}: {r["description"]}' for n, r in self.rooms.items())

    def match_room(self, name):
        if not name or not isinstance(name, str):
            return None
        key = name.strip().lower().replace(' ', '_').replace('-', '_')
        if key in self.rooms:
            return key
        aliases = {n.split('_')[0]: n for n in self.rooms}
        if key in aliases:
            return aliases[key]
        hit = difflib.get_close_matches(key, list(self.rooms), n=1, cutoff=0.6)
        return hit[0] if hit else None

    def interpret(self, instruction):
        prompt = (
            'You control a wheeled robot in an office. The rooms are:\n'
            f'{self.room_list()}\n\n'
            'Tasks:\n'
            '- goto: drive to one of the rooms\n'
            '- find: search for an object and drive to it\n'
            '- follow: find a person and follow them\n\n'
            f'Instruction: "{instruction}"\n'
            'Answer only JSON: {"task": "goto" or "find" or "follow", '
            '"room": a room name or null, "target": the object or person, or null}')
        obj, raw = self.vlm.ask_json(prompt, max_tokens=48, purpose='interpret')
        if not isinstance(obj, dict) or obj.get('task') not in TASKS:
            self.say(f'Anweisung nicht verstanden ({raw!r})')
            return None
        plan = {'task': obj['task'], 'room': self.match_room(obj.get('room')),
                'target': obj.get('target') if isinstance(obj.get('target'), str) else None}
        if plan['task'] == 'goto' and plan['room'] is None:
            self.say(f'unbekannter Raum: {obj.get("room")!r}')
            return None
        if plan['task'] in ('find', 'follow') and not plan['target']:
            plan['target'] = 'person' if plan['task'] == 'follow' else None
            if not plan['target']:
                self.say('kein Suchobjekt erkannt')
                return None
        self.event('plan', **plan)
        return plan

    def rooms_by_distance(self, rooms):
        x, y, _ = self.rover()

        def d(n):
            a = self.rooms[n]['approach']
            return math.hypot(a[0] - x, a[1] - y)
        return sorted(rooms, key=d)

    def rank_rooms(self, target, exclude=()):
        rooms = [r for r in self.rooms if r not in exclude]
        if self.perception == 'oracle':
            return self.rooms_by_distance(rooms)  # Experte ohne Vorwissen: nächster Raum zuerst
        view = render_map_view(self.occ, self.sm, rover=self.rover(), highlight=rooms, done=exclude)
        self.pub_map_view.publish(self.bridge.cv2_to_imgmsg(view, 'bgr8'))
        prompt = (
            'The image is a top-down map of an office. Black = walls and furniture, colored areas '
            '= rooms with their names, the red arrow is the robot.\n'
            f'Rooms:\n{self.room_list()}\n\n'
            f'The robot must find: {target}.\n'
            'In which rooms is it most likely? Answer only JSON: a list of all room names, '
            'most likely first.')
        obj, raw = self.vlm.ask_json(prompt, [view], max_tokens=64, purpose='rank_rooms')
        ranked = []
        for name in obj if isinstance(obj, list) else []:
            r = self.match_room(name if isinstance(name, str) else None)
            if r in rooms and r not in ranked:
                ranked.append(r)
        ranked += self.rooms_by_distance([r for r in rooms if r not in ranked])
        self.event('room_order', target=target, order=ranked)
        return ranked

    def detect(self, snap, target):
        """Box (0–1000) des Ziels im Bild oder None."""
        if self.perception == 'oracle':
            key = oracle_key(target)
            pose = self.gt.get(ORACLE_OBJECTS[key][0]) if key else None
            if pose is None:
                return None
            _, h, r = ORACLE_OBJECTS[key]
            box = oracle_box(snap, pose[:2], h, r)
            self.event('oracle_detect', target=target, box=box)
            return box
        prompt = (f'Is there a {target} in this image? If yes, give its bounding box. Answer only '
                  'JSON: {"found": true, "bbox_2d": [x1, y1, x2, y2]} or {"found": false}')
        obj, _ = self.vlm.ask_json(prompt, [snap.rgb], max_tokens=40, purpose='detect')
        if isinstance(obj, list):  # manchmal kommt eine Liste von Objekten
            obj = obj[0] if obj and isinstance(obj[0], dict) else {}
        if not isinstance(obj, dict):
            return None
        box = obj.get('bbox_2d') or obj.get('bbox')
        if obj.get('found') is False or not valid_box(box):
            return None
        return box

    def verify(self, snap, box, target):
        """Unabhängige Rückfrage am vergrößerten Ausschnitt gegen Fehlalarme.

        (Eine Rückfrage im selben Chat wäre schneller, das 4B-Modell bestätigt
        dort aber meist nur seine erste Antwort.)
        """
        if self.perception == 'oracle':
            return True
        ok, _ = self.vlm.ask_yes_no(f'Is this a {target}?', [crop_box(snap.rgb, box)],
                                    purpose='verify', threshold=VERIFY_THRESHOLD)
        return bool(ok)

    def look_for(self, target, verify=True, snap=None, quiet=False):
        """Im Kamerabild nach dem Ziel suchen → Kartenpunkt (x, y, z) oder None.

        snap: schon aufgenommenes Bild (sonst: anhalten und frisch aufnehmen).
        Die Simulation wird nur pausiert, wenn der Rover steht.
        """
        snap = snap or self.look()
        self.publish_image(annotate(snap.rgb, lines=[f'suche: {target} ...']))
        with self.thinking():
            box = self.detect(snap, target)
            pos = box_to_map(snap, box) if box else None
            cam = snap.T_map_cam[:3, 3]
            if pos is not None:
                dist = math.hypot(pos[0] - cam[0], pos[1] - cam[1])
                if not (0.3 < dist < 9.0 and -0.15 < pos[2] < 2.0
                        and self.occ.clearance(pos[0], pos[1]) > 0):
                    self.event('reject', reason='unplausibler Punkt', box=box, point=pos)
                    pos = None
            if pos is not None and verify and not self.verify(snap, box, target):
                self.event('reject', reason='Rückfrage verneint', box=box)
                self.publish_image(annotate(snap.rgb, [(box, f'{target}?', RED)],
                                            [f'{target}: verworfen']))
                pos = None
        if pos is None:
            if box is None:
                self.publish_image(annotate(snap.rgb, lines=[f'kein {target}']))
            self.event('look', target=target, found=False)
            return None
        self.publish_image(annotate(snap.rgb, [(box, target, GREEN)],
                                    [f'{target}: ({pos[0]:.1f}, {pos[1]:.1f})']))
        self.marks[target] = (pos[0], pos[1], target)
        self.event('look', target=target, found=True, box=box, point=pos)
        if not quiet:
            self.say(f'{target} gesehen bei ({pos[0]:.1f}, {pos[1]:.1f})')
        return pos

    def watch_while_driving(self, target, accept):
        """Hintergrund-Thread: während Nav2 fährt, laufend Bilder an das VLM geben.

        Jedes Bild trägt seine eigene Kamerapose (TF zum Aufnahmezeitpunkt), die
        Lokalisierung stimmt also auch während der Fahrt. Bei einem Treffer, den
        accept(punkt) gutheißt, wird die Fahrt abgebrochen; der Treffer steht dann
        in self.watch_hit.
        """
        self.watch_hit = None
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                if not self.nav.active():  # Ziel noch nicht angenommen oder schon erreicht
                    time.sleep(0.2)
                    continue
                try:
                    snap = self.cam.snapshot(timeout=5.0)
                except RuntimeError:
                    continue
                if stop.is_set():
                    break
                pos = self.look_for(target, snap=snap, quiet=True)
                if pos is not None and accept(pos) and not stop.is_set():
                    self.watch_hit = pos
                    self.say(f'{target} während der Fahrt gesehen bei ({pos[0]:.1f}, {pos[1]:.1f})')
                    self.nav.cancel()
                    break

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

        def finish():
            stop.set()
            thread.join(timeout=60.0)
        return finish

    # ================================================================ Fähigkeiten
    def goto(self, room):
        r = self.rooms[room]
        if not self.enter(room, r['approach'], room):
            return False, f'konnte {room} nicht erreichen'
        if self.perception == 'vlm':
            snap = self.look()
            with self.thinking():
                ok, _ = self.vlm.ask_yes_no(
                    f'A robot was sent to the {room.replace("_", " ")} ({r["description"]}). '
                    'This is its camera image. Does it show that room?', [snap.rgb],
                    purpose='verify_room')
            self.publish_image(annotate(snap.rgb, lines=[f'{room}: {"ja" if ok else "unsicher"}']))
            self.event('room_check', room=room, answer=ok)
        return True, f'im Raum {room} angekommen'

    def search(self, target, rejected=()):
        """Räume nach dem Ziel absuchen → Kartenpunkt oder None.

        rejected: Punkte früherer Fehlalarme, die ignoriert werden.
        """
        self.say(f'suche {target}')

        def accept(p):
            if any(math.hypot(p[0] - r[0], p[1] - r[1]) < 1.0 for r in rejected):
                self.event('reject', reason='bekannter Fehlalarm', point=list(p))
                return False
            return True

        def look():
            p = self.look_for(target)
            return p if p is not None and accept(p) else None
        pos = look()
        if pos is not None:
            return pos
        searched = []
        x, y, _ = self.rover()
        here = room_at(self.sm, x, y)
        for room in self.rank_rooms(target):
            self.say(f'durchsuche {room}')
            for vp in self.rooms[room]['viewpoints']:
                x, y, yaw = self.rover()
                at_vp = (math.hypot(vp[0] - x, vp[1] - y) < 0.4
                         and abs(math.remainder(math.radians(vp[2]) - yaw, 2 * math.pi)) < 0.4)
                if at_vp and room == here and not searched:
                    continue  # von hier aus schon geschaut
                finish = self.watch_while_driving(target, accept)
                try:
                    reached = self.enter(room, vp, f'{room} (Aussichtspunkt)')
                finally:
                    finish()
                hit, self.watch_hit = self.watch_hit, None
                if hit is not None:
                    return hit
                if not reached:
                    continue
                pos = look()
                if pos is not None:
                    return pos
            searched.append(room)
        return None

    def approach(self, xy):
        """Bis APPROACH_DIST vor das Objekt fahren, mit Blick darauf."""
        x, y, _ = self.rover()
        base = math.atan2(y - xy[1], x - xy[0])
        for dang in (0, 25, -25, 50, -50, 80, -80, 110, -110, 150, -150, 180):
            a = base + math.radians(dang)
            for d in (APPROACH_DIST, APPROACH_DIST + 0.4):
                gx, gy, yaw = xy[0] + d * math.cos(a), xy[1] + d * math.sin(a), a + math.pi
                if self.occ.footprint_free(gx, gy, yaw, margin=0.2):
                    if self.go(gx, gy, yaw, 'zum Objekt'):
                        return True
                    break
        return False

    def find(self, target):
        rejected = []
        while True:
            pos = self.search(target, rejected)
            if pos is None:
                return False, f'{target} nicht gefunden'
            if not self.approach(pos):
                return False, f'{target} gefunden bei ({pos[0]:.1f}, {pos[1]:.1f}), aber nicht erreichbar'
            # Abschlussblick aus der Nähe: Liegt die zweite Erkennung (andere Position,
            # anderer Blickwinkel) an derselben Stelle, ist das Objekt bestätigt.
            # Sonst war es ein Fehlalarm – Stelle merken und weitersuchen.
            final = self.look_for(target, verify=False)
            x, y, _ = self.rover()
            if final is not None and math.hypot(final[0] - pos[0], final[1] - pos[1]) < 1.0:
                return True, f'{target} erreicht: {math.hypot(final[0] - x, final[1] - y):.1f} m entfernt'
            self.say(f'aus der Nähe kein {target} – Fehlalarm, suche weiter')
            self.event('false_alarm', point=list(pos))
            rejected.append(pos)
            if len(rejected) >= 3:
                return False, f'{target} nicht gefunden (3 Fehlalarme)'

    def follow(self, target='person', max_time=300.0, start_person=False):
        pos = self.search(target)
        if pos is None:
            return False, f'{target} nicht gefunden'
        cand = self.tracker.nearest_candidate(pos)
        self.tracker.start(cand if cand is not None else pos, self.now())
        self.event('track_start', point=list(pos[:2]))
        if start_person:
            self.pub_person_cmd.publish(String(data='start'))
        self.say(f'folge {target}')
        t_end = self.now() + max_time
        last_goal, last_send, last_log, lost_count, near_done = None, -1e9, -1e9, 0, None
        dists = []
        try:
            while self.now() < t_end:
                time.sleep(0.25)
                now = self.now()
                if self.tracker.lost(now):
                    self.nav.cancel()
                    lost_count += 1
                    self.event('track_lost', count=lost_count)
                    # zuerst mit dem Lidar nahe der letzten Position suchen, dann mit dem VLM
                    last = self.tracker.pos if self.tracker.pos is not None else pos
                    cand = self.tracker.nearest_candidate(last, max_dist=2.0)
                    if cand is not None:
                        self.say(f'Spur verloren ({lost_count}) – Lidar findet die Person wieder')
                        self.tracker.start(cand, self.now())
                        continue
                    self.say(f'Spur verloren ({lost_count}) – schaue mit dem VLM')
                    p = self.look_for(target)
                    if p is None and lost_count >= 3:
                        p = self.search(target)
                    if p is None:
                        if lost_count >= 3:
                            return False, f'{target} verloren'
                        continue
                    cand = self.tracker.nearest_candidate(p)
                    self.tracker.start(cand if cand is not None else p, self.now())
                    last_goal = None
                    continue
                p, v, _ = self.tracker.state(now)
                pos = p
                x, y, _ = self.rover()
                d = math.hypot(p[0] - x, p[1] - y)
                dists.append(d)
                if now - last_log >= 1.0:
                    gt = self.gt.get('person')
                    self.event('follow_state', track=list(p), velocity=list(v), distance=d,
                               gt_person=list(gt[:2]) if gt else None)
                    last_log = now
                goal = None
                if d < HOLD_DIST:
                    self.nav.cancel()
                elif self.nav.active() or d > RESUME_DIST:
                    speed = math.hypot(*v)
                    u = v / speed if speed > 0.15 else (p - np.array([x, y])) / max(d, 1e-3)
                    goal = p - u * FOLLOW_DIST
                    yaw = math.atan2(p[1] - goal[1], p[0] - goal[0])
                    if not self.occ.footprint_free(goal[0], goal[1], yaw, margin=0.1):
                        u = (p - np.array([x, y])) / max(d, 1e-3)
                        goal = p - u * FOLLOW_DIST
                        yaw = math.atan2(u[1], u[0])
                    if (not self.nav.active() or last_goal is None
                            or np.linalg.norm(goal - last_goal) > 0.5 or now - last_send > 3.0):
                        pose = make_pose(goal[0], goal[1], yaw, self.node.get_clock().now().to_msg())
                        self.pub_subgoal.publish(pose)
                        self.nav.send(pose)
                        last_goal, last_send = goal, now
                self.publish_markers(person=p, goal=goal)
                if self.person_state == 'done' and d < 3.5:
                    near_done = near_done or now
                    if now - near_done > 3.0:
                        break
        finally:
            self.nav.cancel()
            self.tracker.stop()
        mean_d = float(np.mean(dists)) if dists else math.nan
        self.event('follow_summary', mean_distance=mean_d, lost=lost_count)
        return True, f'gefolgt, mittlerer Abstand {mean_d:.1f} m, {lost_count}× Spur verloren'

    # ================================================================ Ablauf
    def run(self, instruction, plan=None, meta=None, follow_time=300.0, start_person=False,
            evaluate=None):
        """Eine Anweisung ausführen (als Episode aufgezeichnet). Liefert ein Ergebnis-Dict.

        evaluate(): optionale Auswertung mit Ground Truth (collect.py), landet in meta.json.
        """
        meta = dict(meta or {})
        t0, calls0, secs0 = time.monotonic(), self.vlm.calls, self.vlm.seconds
        self.marks.clear()
        if self.record:
            self.pub_episode.publish(String(data=json.dumps({
                'cmd': 'start', 'instruction': instruction, 'perception': self.perception,
                'model': self.vlm.model if self.perception == 'vlm' else None, **meta})))
            time.sleep(0.3)
        self.say(f'Anweisung: {instruction}')
        success, message = False, ''
        try:
            if plan is None:
                snap_view = self.cam.latest_rgb()
                if snap_view is not None:
                    self.publish_image(annotate(snap_view, lines=[f'"{instruction}"', 'denke nach ...']))
                with self.thinking():
                    plan = self.interpret(instruction)
            if plan is None:
                message = 'Anweisung nicht verstanden'
            else:
                self.say(f'Plan: {plan}')
                if plan['task'] == 'goto':
                    success, message = self.goto(plan['room'])
                elif plan['task'] == 'find':
                    success, message = self.find(plan['target'])
                else:
                    success, message = self.follow(plan['target'] or 'person', follow_time,
                                                   start_person)
        except KeyboardInterrupt:
            self.nav.cancel()
            message = 'abgebrochen'
        finally:
            self.nav.cancel()
        self.say(('✓ ' if success else '✗ ') + message)
        self.publish_markers()
        result = {'success': success, 'message': message, 'plan': plan,
                  'task': plan['task'] if plan else None,
                  'vlm_calls': self.vlm.calls - calls0,
                  'vlm_seconds': round(self.vlm.seconds - secs0, 1),
                  'wall_seconds': round(time.monotonic() - t0, 1)}
        if evaluate:
            result.update(evaluate())
        if self.record:
            self.pub_episode.publish(String(data=json.dumps({'cmd': 'stop', **result})))
        return result

    def shutdown(self):
        self.nav.cancel()
        self.executor.shutdown(timeout_sec=1.0)
        self.cam.node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('instruction', nargs='*', help='z. B. "find the blue bag"')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Ollama-Modell')
    parser.add_argument('--perception', choices=('vlm', 'oracle'), default='vlm')
    parser.add_argument('--no-record', action='store_true', help='keine Episode aufzeichnen')
    parser.add_argument('--no-pause', action='store_true',
                        help='Simulation während VLM-Aufrufen nicht pausieren')
    parser.add_argument('--follow-time', type=float, default=300.0, help='max. Sim-Sekunden folgen')
    parser.add_argument('--start-person', action='store_true',
                        help='beim Folgen die Person loslaufen lassen (/person/command start)')
    parser.add_argument('--reset', action='store_true',
                        help='Rover vorher an die Startpose setzen (Gazebo + AMCL)')
    args = parser.parse_args()

    agent = Agent(args.perception, args.model, record=not args.no_record, pause=not args.no_pause)
    try:
        agent.wait_ready()
        if args.reset:
            agent.say('setze Rover an die Startpose')
            agent.reset_rover()
        if args.instruction:
            res = agent.run(' '.join(args.instruction), follow_time=args.follow_time,
                            start_person=args.start_person)
            return 0 if res['success'] else 1
        print('Befehl eingeben (leer oder Ctrl+D beendet), z. B. "go to the kitchen"')
        while True:
            try:
                line = input('> ').strip()
            except EOFError:
                break
            if not line:
                break
            agent.run(line, follow_time=args.follow_time, start_person=args.start_person)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        agent.shutdown()


if __name__ == '__main__':
    sys.exit(main())
