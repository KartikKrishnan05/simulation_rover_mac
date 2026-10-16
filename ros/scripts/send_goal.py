#!/usr/bin/env python3
"""Schickt den Rover per Nav2 zu benannten Orten im Büro.

  pixi run goal kitchen          # ein Ziel
  pixi run goal tour             # Rundfahrt durch alle Räume
  pixi run goal -- --list        # Orte anzeigen
  pixi run goal 2.0 -1.5 90      # beliebige Pose: x y yaw_grad
"""
import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Name: (x, y, yaw in Grad) – Kartenkoordinaten (= Gazebo-Weltkoordinaten)
# der Hinterachsmitte (base_footprint); die Fahrzeugmitte liegt 0,28 m davor.
PLACES = {
    'start': (1.28, -2.5, 180),
    'desks': (-4.75, -2.7, 180),
    'office_a': (-7.2, 4.0, 90),
    'office_b': (-1.0, 4.0, 90),
    'kitchen': (5.5, 4.6, 0),
    'lounge': (6.0, 1.0, 0),
    # direkt hinter der Tür: im Raum ist kein Platz zum Wenden, zurück geht's rückwärts
    'meeting': (4.1, -2.1, 0),
}
TOUR = ['office_b', 'kitchen', 'lounge', 'meeting', 'desks', 'start']


def make_pose(nav, x, y, yaw_deg):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    yaw = math.radians(float(yaw_deg))
    pose.pose.orientation.z = math.sin(yaw / 2)
    pose.pose.orientation.w = math.cos(yaw / 2)
    return pose


def go(nav, name, x, y, yaw):
    print(f'→ {name} ({x:.2f}, {y:.2f}, {yaw:.0f}°)')
    nav.goToPose(make_pose(nav, x, y, yaw))
    last = None
    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb is not None:
            dist = round(fb.distance_remaining, 1)
            if dist != last:
                print(f'   noch {dist:.1f} m', end='\r', flush=True)
                last = dist
    result = nav.getResult()
    ok = result == TaskResult.SUCCEEDED
    print(f'   {"angekommen" if ok else f"fehlgeschlagen ({result})"}          ')
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', nargs='*', help='Ortsname, "tour" oder x y yaw_grad')
    parser.add_argument('--list', action='store_true', help='bekannte Orte anzeigen')
    args = parser.parse_args()

    if args.list or not args.target:
        for name, (x, y, yaw) in PLACES.items():
            print(f'{name:10s} x={x:6.2f} y={y:6.2f} yaw={yaw:4d}°')
        print('tour       ' + ' → '.join(TOUR))
        return 0

    if len(args.target) == 3:
        goals = [('pose', *map(float, args.target))]
    elif args.target[0] == 'tour':
        goals = [(n, *PLACES[n]) for n in TOUR]
    elif args.target[0] in PLACES:
        goals = [(args.target[0], *PLACES[args.target[0]])]
    else:
        print(f'Unbekanntes Ziel: {" ".join(args.target)} (siehe --list)', file=sys.stderr)
        return 2

    rclpy.init()
    nav = BasicNavigator()
    nav.waitUntilNav2Active()
    ok = all([go(nav, *g) for g in goals])
    nav.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
