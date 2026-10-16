#!/usr/bin/env python3
"""Führt den vorderen und hinteren Lidar zu einem 360°-Scan zusammen.

  /scan + /scan_rear  →  /scan_merged (Frame base_footprint)

AMCL und slam_toolbox können nur einen LaserScan verarbeiten; mit nur dem
vorderen 185°-Scan ist die Lokalisierung in Türen und Gängen mehrdeutig.
Die Punkte beider Scanner werden per TF in base_footprint transformiert und
in Winkel-Bins einsortiert (pro Bin gewinnt der nächste Punkt).
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

BINS = 720
RANGE_MAX = 12.0


class ScanMerger(Node):
    def __init__(self):
        super().__init__('scan_merger')
        self.frame = self.declare_parameter('target_frame', 'base_footprint').value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.mounts = {}  # Sensor-Frame → (x, y, yaw) in target_frame (statisch)
        self.rear = None
        self.pub = self.create_publisher(LaserScan, 'scan_merged', qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.on_front, qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan_rear', self.on_rear, qos_profile_sensor_data)

    def mount(self, frame):
        if frame not in self.mounts:
            t = self.tf_buffer.lookup_transform(self.frame, frame, rclpy.time.Time())
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.mounts[frame] = (t.transform.translation.x, t.transform.translation.y, yaw)
        return self.mounts[frame]

    def on_rear(self, msg):
        self.rear = msg

    def add_points(self, scan, bins):
        mx, my, myaw = self.mount(scan.header.frame_id)
        step = 2 * math.pi / BINS
        a = scan.angle_min
        for r in scan.ranges:
            if scan.range_min <= r <= scan.range_max and math.isfinite(r):
                x = mx + r * math.cos(myaw + a)
                y = my + r * math.sin(myaw + a)
                d = math.hypot(x, y)
                k = int((math.atan2(y, x) + math.pi) / step) % BINS
                if d < bins[k]:
                    bins[k] = d
            a += scan.angle_increment

    def on_front(self, msg):
        bins = [math.inf] * BINS
        try:
            self.add_points(msg, bins)
            if self.rear is not None:
                self.add_points(self.rear, bins)
        except TransformException as e:
            self.get_logger().warn(f'TF noch nicht verfügbar: {e}', throttle_duration_sec=5.0)
            return
        out = LaserScan()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame
        out.angle_min = -math.pi
        out.angle_max = math.pi - 2 * math.pi / BINS
        out.angle_increment = 2 * math.pi / BINS
        out.scan_time = msg.scan_time
        out.range_min = 0.05
        out.range_max = RANGE_MAX
        out.ranges = bins
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
