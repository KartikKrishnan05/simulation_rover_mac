"""Verfolgt eine Person mit den beiden Lidaren (10 Hz).

Das VLM ist mit 2–8 s pro Antwort zu langsam für einen Regelkreis. Es
liefert nur den Startpunkt (wo im Bild ist die Person?) und hilft, wenn die
Spur verloren geht. Dazwischen arbeitet dieser Tracker:

  1. Scanpunkte per TF in den Frame map transformieren
  2. Punkte auf der statischen Karte verwerfen (Wände, Möbel)
  3. übrige Punkte zu Clustern gruppieren (Beine einer Person: ~0,5 m)
  4. Cluster nahe der vorhergesagten Position übernehmen (Gating),
     Position/Geschwindigkeit glätten
"""
import math
import threading

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException

from common import yaw_of

STATIC_CLEARANCE = 0.25   # m: Punkte so nah an der Karte gehören zur Umgebung
CLUSTER_GAP = 0.25        # m zwischen Nachbarpunkten eines Clusters
MERGE_DIST = 0.55         # m: zwei Beine → eine Person
MAX_WIDTH = 0.9           # m: breitere Cluster sind keine Person
GATE = 1.0                # m um die vorhergesagte Position
LOST_AFTER = 4.0          # s ohne Messung
MAX_SPEED = 1.0           # m/s – mehr schafft die Person nicht (Ausreißer beim Abbiegen)


class PersonTracker:
    def __init__(self, node, tf_buffer, occ):
        self.node = node
        self.tf = tf_buffer
        self.occ = occ
        self.lock = threading.Lock()
        self.pos = None
        self.vel = np.zeros(2)
        self.t_seen = None
        self.t_last = None
        self.candidates = {}  # Lidar-Frame → letzte Cluster (für die (Wieder-)Initialisierung)
        node.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        node.create_subscription(LaserScan, '/scan_rear', self._on_scan, qos_profile_sensor_data)

    def start(self, xy, t):
        with self.lock:
            self.pos = np.array(xy[:2], dtype=float)
            self.vel = np.zeros(2)
            self.t_seen = self.t_last = t

    def stop(self):
        with self.lock:
            self.pos = None

    def state(self, now):
        """(Position, Geschwindigkeit, Sekunden seit letzter Messung) oder None."""
        with self.lock:
            if self.pos is None:
                return None
            return self.pos.copy(), self.vel.copy(), now - self.t_seen

    def lost(self, now):
        s = self.state(now)
        return s is None or s[2] > LOST_AFTER

    def _clusters(self, msg):
        try:
            tf = self.tf.lookup_transform('map', msg.header.frame_id, Time())
        except TransformException:
            return []
        tr, yaw = tf.transform.translation, yaw_of(tf.transform.rotation)
        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < min(msg.range_max, 8.0))
        r, a = r[ok], a[ok]
        xs = tr.x + r * np.cos(yaw + a)
        ys = tr.y + r * np.sin(yaw + a)
        keep = np.array([self.occ.clearance(x, y) > STATIC_CLEARANCE for x, y in zip(xs, ys)],
                        dtype=bool)
        pts = np.stack([xs, ys], axis=1)
        clusters, cur, prev_i = [], [], None
        for i in np.flatnonzero(keep):
            if cur and (i != prev_i + 1 or np.linalg.norm(pts[i] - pts[prev_i]) > CLUSTER_GAP):
                clusters.append(np.array(cur))
                cur = []
            cur.append(pts[i])
            prev_i = i
        if cur:
            clusters.append(np.array(cur))
        cents = []
        for c in clusters:
            if len(c) >= 2 and np.linalg.norm(c.max(axis=0) - c.min(axis=0)) < MAX_WIDTH:
                cents.append(c.mean(axis=0))
        merged = []  # Beine zusammenfassen
        for c in cents:
            for m in merged:
                if np.linalg.norm(m[0] / m[1] - c) < MERGE_DIST:
                    m[0] += c
                    m[1] += 1
                    break
            else:
                merged.append([c.copy(), 1])
        return [m[0] / m[1] for m in merged]

    def _on_scan(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cands = self._clusters(msg)
        with self.lock:
            self.candidates[msg.header.frame_id] = cands
            if self.pos is None or not cands:
                return
            dt = max(0.0, min(1.0, t - self.t_last)) if self.t_last else 0.0
            pred = self.pos + self.vel * dt
            d = [np.linalg.norm(c - pred) for c in cands]
            k = int(np.argmin(d))
            if d[k] > GATE + 0.5 * math.hypot(*self.vel) * max(0.0, t - self.t_seen):
                return
            meas = cands[k]
            if dt > 1e-3:
                self.vel = 0.85 * self.vel + 0.15 * (meas - self.pos) / max(dt, 0.05)
                speed = math.hypot(*self.vel)
                if speed > MAX_SPEED:
                    self.vel *= MAX_SPEED / speed
            self.pos = 0.3 * pred + 0.7 * meas
            self.t_seen = self.t_last = t

    def nearest_candidate(self, xy, max_dist=1.2):
        with self.lock:
            cands = [c for cs in self.candidates.values() for c in cs]
        if not cands:
            return None
        d = [np.linalg.norm(c - np.asarray(xy[:2])) for c in cands]
        k = int(np.argmin(d))
        return cands[k] if d[k] <= max_dist else None
