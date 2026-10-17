"""Gemeinsame Bausteine der VLM-Szenarien.

  - semantische Karte (ros/config/semantic_map.yaml) und Belegungskarte
  - Geometrie-Helfer (Yaw ↔ Quaternion, Posen)
  - SimControl: Simulation pausieren, Objekte versetzen (über ros_gz_bridge)
  - Navigator: schlanker NavigateToPose-Client (Ziele lassen sich jederzeit
    ersetzen – das braucht das Folgen)
  - GroundTruth: echte Posen aus Gazebo (Auswertung, Orakel)

Alle Klassen erwarten einen Node, der in einem Hintergrund-Thread gespinnt
wird (siehe agent.py); Methoden blockieren per Polling auf Futures.
"""
import math
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from rclpy.action import ActionClient
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from tf2_msgs.msg import TFMessage

ROS_DIR = Path(__file__).resolve().parent.parent
ROOT = ROS_DIR.parent
SEMANTIC_MAP = ROS_DIR / 'config' / 'semantic_map.yaml'
MAP_YAML = ROS_DIR / 'maps' / 'office.yaml'
WORLD = 'office'


def load_semantic_map(path=SEMANTIC_MAP):
    with open(path) as f:
        return yaml.safe_load(f)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def make_pose(x, y, yaw, stamp=None, frame='map'):
    p = PoseStamped()
    p.header.frame_id = frame
    if stamp is not None:
        p.header.stamp = stamp
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.orientation.z = math.sin(yaw / 2)
    p.pose.orientation.w = math.cos(yaw / 2)
    return p


def wait_future(fut, timeout):
    t0 = time.monotonic()
    while not fut.done():
        if time.monotonic() - t0 > timeout:
            return None
        time.sleep(0.01)
    return fut.result()


def in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def room_at(sm, x, y):
    for name, room in sm['rooms'].items():
        if in_rect(x, y, room['rect']):
            return name
    return None


class OccupancyMap:
    """Die statische Nav2-Karte mit Abstandsfeld (m bis zum nächsten Hindernis)."""

    def __init__(self, yaml_path=MAP_YAML):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img = cv2.imread(str(Path(yaml_path).parent / meta['image']), cv2.IMREAD_GRAYSCALE)
        self.res = float(meta['resolution'])
        self.x0, self.y0 = meta['origin'][:2]
        self.h, self.w = img.shape
        self.image = img
        free = (img > 127).astype(np.uint8)
        self.dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.res

    def to_px(self, x, y):
        return int((x - self.x0) / self.res), int(self.h - 1 - (y - self.y0) / self.res)

    def clearance(self, x, y):
        i, j = self.to_px(x, y)
        if not (0 <= i < self.w and 0 <= j < self.h):
            return 0.0
        return float(self.dist[j, i])

    def footprint_clearance(self, x, y, yaw):
        """Kleinster Abstand des Rover-Footprints (wie nav2_params.yaml) zur Karte."""
        c, s = math.cos(yaw), math.sin(yaw)
        return min(self.clearance(x + c * fx - s * fy, y + s * fx + c * fy)
                   for fx in np.linspace(-0.17, 0.84, 6) for fy in (-0.38, 0.0, 0.38))

    def footprint_free(self, x, y, yaw, margin=0.15):
        return self.footprint_clearance(x, y, yaw) >= margin


class SimControl:
    """Gazebo über die ros_gz_bridge-Services steuern."""

    def __init__(self, node):
        self.node = node
        self.control = node.create_client(ControlWorld, f'/world/{WORLD}/control')
        self.set_pose_cli = node.create_client(SetEntityPose, f'/world/{WORLD}/set_pose')
        self.enabled = True

    def pause(self, on):
        if not self.enabled:
            return False
        req = ControlWorld.Request()
        req.world_control.pause = bool(on)
        if not self.control.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().warn('Service /world/office/control fehlt – pausiere nicht')
            self.enabled = False
            return False
        res = wait_future(self.control.call_async(req), 5.0)
        return bool(res and res.success)

    @contextmanager
    def paused(self, on=True):
        """Simulation anhalten, solange das VLM nachdenkt."""
        did = on and self.pause(True)
        try:
            yield
        finally:
            if did:
                self.pause(False)

    def set_pose_async(self, name, x, y, yaw, z=0.0):
        """Objekt versetzen ohne zu warten (für Nodes, die nur einen Thread spinnen)."""
        req = SetEntityPose.Request()
        req.entity = Entity(name=name, type=Entity.MODEL)
        req.pose.position.x, req.pose.position.y, req.pose.position.z = float(x), float(y), float(z)
        req.pose.orientation.z = math.sin(yaw / 2)
        req.pose.orientation.w = math.cos(yaw / 2)
        return self.set_pose_cli.call_async(req)

    def set_pose(self, name, x, y, yaw, z=0.0):
        if not self.set_pose_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Service /world/office/set_pose nicht erreichbar')
        res = wait_future(self.set_pose_async(name, x, y, yaw, z), 5.0)
        return bool(res and res.success)


class GroundTruth:
    """Echte Posen (x, y, yaw) von Rover, Person und Tasche aus Gazebo."""

    TOPICS = {'rover': '/rover/ground_truth', 'person': '/person/ground_truth',
              'bag': '/bag/ground_truth'}

    def __init__(self, node):
        self.poses = {}
        for name, topic in self.TOPICS.items():
            node.create_subscription(TFMessage, topic,
                                     lambda m, n=name: self._on(n, m), 10)

    def _on(self, name, msg):
        if msg.transforms:
            t = msg.transforms[0].transform
            self.poses[name] = (t.translation.x, t.translation.y, yaw_of(t.rotation))

    def get(self, name):
        return self.poses.get(name)


class Navigator:
    """Nav2-Client; ein neues Ziel ersetzt das laufende (Preemption).

    send(pose)            NavigateToPose
    send_through(poses)   NavigateThroughPoses – ein durchgehender Pfad, z. B.
                          gerade durch eine Tür (Zwischenposen werden nicht
                          einzeln angehalten und ausgerichtet)
    """

    def __init__(self, node):
        self.node = node
        self.client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
        self.through_client = ActionClient(node, NavigateThroughPoses, 'navigate_through_poses')
        self.handle = None
        self.result = None
        self.goal = None
        self.feedback = None

    def send(self, pose):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        return self._send(self.client, goal, pose)

    def send_through(self, poses):
        goal = NavigateThroughPoses.Goal()
        goal.poses = list(poses)
        return self._send(self.through_client, goal, poses[-1])

    def _send(self, client, goal, final_pose):
        if not client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError('Nav2-Action-Server nicht erreichbar')
        handle = wait_future(client.send_goal_async(goal, self._on_feedback), 10.0)
        if handle is None or not handle.accepted:
            return False
        self.handle, self.goal = handle, final_pose
        self.result = handle.get_result_async()
        return True

    def _on_feedback(self, msg):
        self.feedback = msg.feedback

    def active(self):
        return self.result is not None and not self.result.done()

    def status(self):
        if self.result is None or not self.result.done():
            return None
        return self.result.result().status

    def wait(self, timeout=600.0, tick=None):
        """Bis zum Ende warten; tick() wird regelmäßig aufgerufen.

        timeout zählt in Sim-Zeit: Läuft die Simulation langsamer als Echtzeit
        (Mac im Stromsparmodus/ausgelastet), bricht die Fahrt sonst zu früh ab.
        Die Wanduhr-Grenze (10 × timeout) fängt eine stehende Sim-Uhr ab.
        """
        clock = self.node.get_clock()
        t0, w0 = clock.now().nanoseconds * 1e-9, time.monotonic()
        while self.active():
            if (clock.now().nanoseconds * 1e-9 - t0 > timeout
                    or time.monotonic() - w0 > 10 * timeout):
                self.cancel()
                return False
            if tick:
                tick()
            time.sleep(0.1)
        return self.status() == GoalStatus.STATUS_SUCCEEDED

    def cancel(self):
        if self.active():
            wait_future(self.handle.cancel_goal_async(), 5.0)
            t0 = time.monotonic()
            while self.active() and time.monotonic() - t0 < 5.0:
                time.sleep(0.05)
