#!/usr/bin/env python3
"""Zeichnet Episoden als Trainingsdaten für ein späteres VLA auf.

Gesteuert über /episode/control (std_msgs/String mit JSON):
  {"cmd": "start", "instruction": "find the blue bag", "task": "find", ...}
  {"cmd": "stop", "success": true, ...}
Alle weiteren Felder landen in meta.json. Der Agent (agent.py) startet und
beendet Episoden selbst; aufgezeichnet wird auch, wenn ein Mensch per
Teleop fährt.

Pro Schritt (Standard 10 Hz Sim-Zeit; in der Pause ruht auch die Aufnahme):

  data/episodes/<id>/
    meta.json        Anweisung, Aufgabe, Erfolg, Modell, Ground Truth …
    steps.npz        t                 Sim-Zeit (s)
                     state             [x, y, yaw] aus AMCL (map → base_footprint)
                     velocity          [v, ω] gemessen (/odom)
                     steering          mittlerer Lenkwinkel der Vorderräder (rad)
                     action            [v, ω] befohlen (/cmd_vel, das Expertenlabel)
                     subgoal           [x, y, yaw] aktuelles Nav2-Ziel (NaN = keins)
                     ground_truth      [x, y, yaw] Rover laut Gazebo (Fahrzeugmitte)
                     scan              360°-Scan /scan_merged (720 Werte, float16)
    rgb/000000.jpg   Kamera 640×480
    depth/000000.png Tiefe 320×240, uint16 in mm (0 = ungültig)
    events.jsonl     VLM-Aufrufe und Agenten-Entscheidungen mit Schrittindex

Die Struktur lässt sich direkt in das LeRobot-Format überführen
(observation.images.front, observation.state, action, task).
"""
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, yaw_of  # noqa: E402

NAN3 = [math.nan] * 3


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__('episode_recorder')
        self.rate = self.declare_parameter('rate', 10.0).value
        self.out_dir = Path(self.declare_parameter('out_dir', str(ROOT / 'data' / 'episodes')).value)
        self.bridge = CvBridge()
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.latest = {}
        self.cmd_time = None
        self.episode = None
        # Bilder nur während einer Episode abonnieren (die Bridge konvertiert lazy)
        self.image_topics = [(Image, '/camera/image_raw', 'rgb'), (Image, '/camera/depth', 'depth')]
        self.image_subs = []
        subs = [
            (LaserScan, '/scan_merged', 'scan', qos_profile_sensor_data),
            (Odometry, '/odom', 'odom', 10),
            (JointState, '/joint_states', 'joints', qos_profile_sensor_data),
            (PoseStamped, '/vlm/subgoal', 'subgoal', 10),
            (TFMessage, '/rover/ground_truth', 'gt', 10),
        ]
        for typ, topic, key, qos in subs:
            self.create_subscription(typ, topic, lambda m, k=key: self.latest.__setitem__(k, m), qos)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(String, '/episode/control', self.on_control, 10)
        self.create_subscription(String, '/vlm/events', self.on_event, 50)
        self.create_timer(1.0 / self.rate, self.record)
        self.get_logger().info(f'bereit, schreibe nach {self.out_dir}')

    def on_cmd(self, msg):
        self.latest['cmd'] = msg
        self.cmd_time = self.now()

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ Steuerung
    def on_control(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f'kein JSON: {msg.data!r}')
            return
        cmd = data.pop('cmd', None)
        if cmd == 'start':
            if self.episode:
                self.finish({'success': None, 'note': 'abgebrochen durch neue Episode'})
            eid = data.pop('episode_id', None) or time.strftime('%Y%m%d_%H%M%S')
            path = self.out_dir / eid
            (path / 'rgb').mkdir(parents=True, exist_ok=True)
            (path / 'depth').mkdir(exist_ok=True)
            self.image_subs = [
                self.create_subscription(typ, topic, lambda m, k=key: self.latest.__setitem__(k, m),
                                         qos_profile_sensor_data)
                for typ, topic, key in self.image_topics]
            self.episode = {'id': eid, 'path': path, 'meta': data, 'steps': [],
                            'events': open(path / 'events.jsonl', 'w'),
                            'start_wall': time.time(), 'start_sim': self.now()}
            self.get_logger().info(f'Episode {eid} gestartet: {data.get("instruction", "")}')
        elif cmd == 'stop' and self.episode:
            self.finish(data)
        elif cmd == 'discard' and self.episode:
            path = self.episode['path']
            self.close_images()
            self.episode['events'].close()
            self.episode = None
            shutil.rmtree(path, ignore_errors=True)
            self.get_logger().info(f'Episode verworfen: {path.name}')

    def on_event(self, msg):
        if self.episode:
            try:
                ev = json.loads(msg.data)
            except json.JSONDecodeError:
                ev = {'text': msg.data}
            ev['step'] = len(self.episode['steps'])
            ev['t'] = round(self.now(), 3)
            self.episode['events'].write(json.dumps(ev) + '\n')
            self.episode['events'].flush()

    # ------------------------------------------------------------ Aufnahme
    def state(self):
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', Time())
        except TransformException:
            return NAN3
        tr = t.transform.translation
        return [tr.x, tr.y, yaw_of(t.transform.rotation)]

    def record(self):
        ep = self.episode
        if ep is None or 'rgb' not in self.latest:
            return
        i = len(ep['steps'])
        L = self.latest
        rgb = self.bridge.imgmsg_to_cv2(L['rgb'], 'bgr8')
        cv2.imwrite(str(ep['path'] / 'rgb' / f'{i:06d}.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if 'depth' in L:
            d = self.bridge.imgmsg_to_cv2(L['depth'], 'passthrough').astype(np.float32)
            d = np.where(np.isfinite(d), np.clip(d * 1000, 0, 65535), 0).astype(np.uint16)
            cv2.imwrite(str(ep['path'] / 'depth' / f'{i:06d}.png'), d)
        cmd = L.get('cmd') if self.cmd_time and self.now() - self.cmd_time < 0.5 else None
        odom = L.get('odom')
        steer = math.nan
        if 'joints' in L:
            js = dict(zip(L['joints'].name, L['joints'].position))
            vals = [js[k] for k in ('front_left_steering_joint', 'front_right_steering_joint') if k in js]
            steer = sum(vals) / len(vals) if vals else math.nan
        sg = L.get('subgoal')
        gt = L.get('gt')
        ep['steps'].append({
            't': self.now(),
            'state': self.state(),
            'velocity': [odom.twist.twist.linear.x, odom.twist.twist.angular.z] if odom else [math.nan] * 2,
            'steering': steer,
            'action': [cmd.linear.x, cmd.angular.z] if cmd else [0.0, 0.0],
            'subgoal': ([sg.pose.position.x, sg.pose.position.y, yaw_of(sg.pose.orientation)]
                        if sg and sg.header.frame_id else NAN3),
            'ground_truth': ([gt.transforms[0].transform.translation.x,
                              gt.transforms[0].transform.translation.y,
                              yaw_of(gt.transforms[0].transform.rotation)]
                             if gt and gt.transforms else NAN3),
            'scan': (np.nan_to_num(np.asarray(L['scan'].ranges, dtype=np.float32), posinf=0.0)
                     if 'scan' in L else np.zeros(720, np.float32)),
        })

    def close_images(self):
        for sub in self.image_subs:
            self.destroy_subscription(sub)
        self.image_subs = []
        self.latest.pop('rgb', None)
        self.latest.pop('depth', None)

    def finish(self, result):
        self.close_images()
        ep, self.episode = self.episode, None
        ep['events'].close()
        steps = ep['steps']
        arrays = {k: np.array([s[k] for s in steps], dtype=np.float64)
                  for k in ('t', 'state', 'velocity', 'steering', 'action', 'subgoal', 'ground_truth')}
        if steps:
            arrays['scan'] = np.stack([s['scan'] for s in steps]).astype(np.float16)
        np.savez_compressed(ep['path'] / 'steps.npz', **arrays)
        meta = {**ep['meta'], **result, 'episode_id': ep['id'], 'num_steps': len(steps),
                'rate_hz': self.rate, 'sim_duration_s': round(self.now() - ep['start_sim'], 2),
                'wall_duration_s': round(time.time() - ep['start_wall'], 2)}
        (ep['path'] / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        self.get_logger().info(f'Episode {ep["id"]} gespeichert: {len(steps)} Schritte, '
                               f'Erfolg={result.get("success")}')


def main():
    rclpy.init()
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.episode:
            node.finish({'success': None, 'note': 'Recorder beendet'})


if __name__ == '__main__':
    main()
