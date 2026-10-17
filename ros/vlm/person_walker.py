#!/usr/bin/env python3
"""Lässt die Person in Gazebo ihren festen Weg ablaufen.

Weg und Tempo: person_path in ros/config/semantic_map.yaml. Die Person
wartet am Start, bis sie ein Kommando bekommt:

  ros2 topic pub --once /person/command std_msgs/String "data: start"
  … "data: reset"   zurück an den Start (per set_pose) und warten
  … "data: park"    an den Parkplatz abseits der Fahrwege (person_path.park)
  … "data: stop"    stehen bleiben

Status auf /person/state: idle | walking | done. Geregelt wird mit einem
einfachen Pure-Pursuit auf die Ground-Truth-Pose (/person/ground_truth);
der Fahrbefehl geht über die Bridge an das VelocityControl-System.
"""
import math
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SimControl, load_semantic_map, yaw_of  # noqa: E402

REACHED = 0.25  # m


class PersonWalker(Node):
    def __init__(self):
        super().__init__('person_walker')
        path = load_semantic_map()['person_path']
        self.speed = float(path['speed'])
        self.park = tuple(map(float, path['park']))
        self.waypoints = [tuple(map(float, w)) for w in path['waypoints']]
        self.sim = SimControl(self)
        self.pose = None
        self.state = 'idle'
        self.index = 1
        self.wait_until = None
        self.pending_reset = None
        self.cmd = self.create_publisher(Twist, '/person/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/person/state', 10)
        self.create_subscription(TFMessage, '/person/ground_truth', self.on_pose, 10)
        self.create_subscription(String, '/person/command', self.on_command, 10)
        self.create_timer(0.05, self.step)
        self.create_timer(0.5, lambda: self.state_pub.publish(String(data=self.state)))

    def on_pose(self, msg):
        if msg.transforms:
            t = msg.transforms[0].transform
            self.pose = (t.translation.x, t.translation.y, yaw_of(t.rotation))

    def on_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            if self.state != 'walking':
                self.index, self.wait_until = 1, None
            self.state = 'walking'
        elif cmd == 'reset':
            self.state = 'idle'
            self.cmd.publish(Twist())
            (x0, y0, _), (x1, y1, _) = self.waypoints[:2]
            # Service-Aufruf nicht im Callback blockieren: im Timer erledigen
            self.pending_reset = (x0, y0, math.atan2(y1 - y0, x1 - x0))
        elif cmd == 'park':
            self.state = 'idle'
            self.cmd.publish(Twist())
            self.pending_reset = (*self.park, 0.0)
        elif cmd == 'stop':
            self.state = 'idle'
            self.cmd.publish(Twist())
        self.get_logger().info(f'Kommando {cmd!r} → {self.state}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def step(self):
        reset = self.pending_reset
        if reset is not None:
            self.pending_reset = None
            self.sim.set_pose_async('person', *reset, z=0.01)
            return
        if self.state != 'walking' or self.pose is None:
            return
        x, y, yaw = self.pose
        tx, ty, pause = self.waypoints[self.index]
        if self.wait_until is not None:
            if self.now() < self.wait_until:
                self.cmd.publish(Twist())
                return
            self.wait_until = None
            self.advance()
            return
        dist = math.hypot(tx - x, ty - y)
        if dist < REACHED:
            if pause > 0:
                self.wait_until = self.now() + pause
                self.cmd.publish(Twist())
            else:
                self.advance()
            return
        err = math.atan2(ty - y, tx - x) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        cmd = Twist()
        cmd.angular.z = max(-1.5, min(1.5, 2.5 * err))
        cmd.linear.x = self.speed * max(0.0, math.cos(err)) ** 4 * min(1.0, dist / 0.4 + 0.3)
        self.cmd.publish(cmd)

    def advance(self):
        self.index += 1
        if self.index >= len(self.waypoints):
            self.state = 'done'
            self.cmd.publish(Twist())
            self.get_logger().info('Ziel erreicht')


def main():
    rclpy.init()
    node = PersonWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
