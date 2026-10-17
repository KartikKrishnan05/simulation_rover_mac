"""Kamera, Tiefe und Projektion: VLM-Bounding-Box → Punkt auf der Karte.

  - Camera.snapshot(): Farbbild + Tiefenbild + Kamerapose (TF map → Kamera)
  - box_to_map():      Box (0–1000, Qwen3-VL) → 3D-Punkt im Frame map
  - oracle_box():      Box eines bekannten Objekts, falls sichtbar (Ground
                       Truth; ersetzt beim Datensammeln das VLM)
  - annotate(), render_map_view(): Bilder für RViz und den VLM-Prompt
"""
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformException

CAMERA_FRAME = 'camera_optical_frame'


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class Snapshot:
    rgb: np.ndarray        # BGR, 480×640
    depth: np.ndarray      # float32 in m, 240×320 (inf = nichts getroffen)
    K: np.ndarray          # Intrinsik der Farbkamera
    T_map_cam: np.ndarray  # 4×4
    stamp: float           # Sim-Zeit in s

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]


class Camera:
    """Abonniert die Bilder nur, solange sie gebraucht werden: Die Bridge
    konvertiert sie dann nur bei Bedarf (lazy), das entlastet den Rechner.

    Eigener Node, den nur snapshot() spinnt – so lassen sich die Abos
    gefahrlos anlegen und wieder entfernen.
    """

    def __init__(self, node, tf_buffer):
        self.node = rclpy.create_node(node.get_name() + '_camera')
        self.tf = tf_buffer
        self.bridge = CvBridge()
        self.rgb = self.depth = self.info = None
        self.wait_seconds = 0.0

    def _on_rgb(self, m):
        self.rgb = m

    def _on_depth(self, m):
        self.depth = m

    def _on_info(self, m):
        # nur die Intrinsik der Farbkamera (640×480) übernehmen
        if m.width == 640 and m.height == 480:
            self.info = m

    @staticmethod
    def _t(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def latest_rgb(self):
        return None if self.rgb is None else self.bridge.imgmsg_to_cv2(self.rgb, 'bgr8')

    def snapshot(self, after=None, timeout=20.0):
        """Neueste Bilder, beide aufgenommen nach Sim-Zeit `after` (s)."""
        n, q = self.node, qos_profile_sensor_data
        self.rgb = self.depth = None  # nie ein altes Bild von einer anderen Pose verwenden
        subs = [n.create_subscription(Image, '/camera/image_raw', self._on_rgb, q),
                n.create_subscription(Image, '/camera/depth', self._on_depth, q),
                n.create_subscription(CameraInfo, '/camera/camera_info', self._on_info, q)]
        try:
            t0 = time.monotonic()
            while True:
                rclpy.spin_once(n, timeout_sec=0.05)
                ok = self.rgb is not None and self.depth is not None and self.info is not None
                if ok and (after is None or min(self._t(self.rgb), self._t(self.depth)) > after):
                    break
                if time.monotonic() - t0 > timeout:
                    raise RuntimeError('keine aktuellen Kamerabilder (/camera/image_raw, /camera/depth)')
            self.wait_seconds = time.monotonic() - t0
            return self._convert()
        finally:
            for sub in subs:
                n.destroy_subscription(sub)

    def _convert(self):
        rgb_msg = self.rgb
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(self.depth, 'passthrough').astype(np.float32)
        K = np.array(self.info.k, dtype=float).reshape(3, 3)
        try:
            tf = self.tf.lookup_transform('map', CAMERA_FRAME, Time.from_msg(rgb_msg.header.stamp),
                                          timeout=Duration(seconds=1.0))
        except TransformException:
            tf = self.tf.lookup_transform('map', CAMERA_FRAME, Time())
        T = np.eye(4)
        T[:3, :3] = quat_matrix(tf.transform.rotation)
        tr = tf.transform.translation
        T[:3, 3] = (tr.x, tr.y, tr.z)
        return Snapshot(rgb, depth, K, T, self._t(rgb_msg))


def box_to_pixels(box, size):
    """Box [x1, y1, x2, y2] (0–1000) → Pixel, sortiert und auf das Bild begrenzt."""
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (int(np.clip(x1 / 1000 * w, 0, w - 1)), int(np.clip(y1 / 1000 * h, 0, h - 1)),
            int(np.clip(x2 / 1000 * w, 0, w - 1)), int(np.clip(y2 / 1000 * h, 0, h - 1)))


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
            and abs(box[2] - box[0]) >= 3 and abs(box[3] - box[1]) >= 3)


def box_to_map(snap, box):
    """3D-Punkt (x, y, z) im Frame map für die Box – oder None.

    Nimmt die innere Hälfte der Box im Tiefenbild, sucht die nächste Fläche
    (20%-Perzentil) und mittelt die Punkte bis 0,35 m dahinter. So landet
    der Punkt auf dem Objekt und nicht auf der Wand hinter ihm.
    """
    x1, y1, x2, y2 = box_to_pixels(box, snap.size)
    sx = snap.depth.shape[1] / snap.size[0]
    sy = snap.depth.shape[0] / snap.size[1]
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
    u0, u1 = int((cx - bw) * sx), int(math.ceil((cx + bw) * sx)) + 1
    v0, v1 = int((cy - bh) * sy), int(math.ceil((cy + bh) * sy)) + 1
    patch = snap.depth[v0:v1, u0:u1]
    vs, us = np.mgrid[v0:v1, u0:u1]
    ok = np.isfinite(patch) & (patch > 0.1) & (patch < 11.5)
    if ok.sum() < 2:
        return None
    near = np.percentile(patch[ok], 20)
    sel = ok & (patch <= near + 0.35)
    z = patch[sel]
    fx, fy = snap.K[0, 0] * sx, snap.K[1, 1] * sy
    ppx, ppy = snap.K[0, 2] * sx, snap.K[1, 2] * sy
    pts = np.stack([(us[sel] + 0.5 - ppx) * z / fx, (vs[sel] + 0.5 - ppy) * z / fy, z, np.ones_like(z)])
    p = (snap.T_map_cam @ pts).mean(axis=1)
    return float(p[0]), float(p[1]), float(p[2])


def project(snap, p_map):
    """Kartenpunkt → (u, v, Tiefe) im Farbbild."""
    q = np.linalg.inv(snap.T_map_cam) @ np.array([*p_map, 1.0])
    if q[2] <= 0.05:
        return None
    u = snap.K[0, 0] * q[0] / q[2] + snap.K[0, 2]
    v = snap.K[1, 1] * q[1] / q[2] + snap.K[1, 2]
    return u, v, q[2]


def oracle_box(snap, xy, height, radius, max_range=7.0):
    """Box (0–1000) eines Objekts mit bekannter Position, wenn es im Bild zu sehen ist."""
    w, h = snap.size
    c = project(snap, (xy[0], xy[1], height / 2))
    if c is None or not (0 < c[0] < w and 0 < c[1] < h) or c[2] > max_range:
        return None
    # verdeckt? Tiefenbild an der Objektmitte mit der erwarteten Tiefe vergleichen
    sx = snap.depth.shape[1] / w
    sy = snap.depth.shape[0] / h
    u, v = int(c[0] * sx), int(c[1] * sy)
    patch = snap.depth[max(0, v - 1):v + 2, max(0, u - 1):u + 2]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0 or np.median(patch) < c[2] - radius - 0.3:
        return None
    corners = [project(snap, (xy[0] + dx, xy[1] + dy, z))
               for dx in (-radius, radius) for dy in (-radius, radius) for z in (0.0, height)]
    corners = [p for p in corners if p is not None]
    us = np.clip([p[0] for p in corners], 0, w - 1)
    vs = np.clip([p[1] for p in corners], 0, h - 1)
    if us.max() - us.min() < 4 or vs.max() - vs.min() < 4:
        return None
    return [int(us.min() / w * 1000), int(vs.min() / h * 1000),
            int(us.max() / w * 1000), int(vs.max() / h * 1000)]


def crop_box(img, box, margin=0.6, min_side=224):
    """Ausschnitt um die Box, vergrößert – für die unabhängige Rückfrage an das VLM."""
    x1, y1, x2, y2 = box_to_pixels(box, (img.shape[1], img.shape[0]))
    mw, mh = (x2 - x1) * margin + 10, (y2 - y1) * margin + 10
    x1, x2 = int(max(0, x1 - mw)), int(min(img.shape[1], x2 + mw))
    y1, y2 = int(max(0, y1 - mh)), int(min(img.shape[0], y2 + mh))
    crop = img[y1:y2, x1:x2]
    s = max(1.0, min_side / max(1, min(crop.shape[:2])))
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def annotate(img, boxes=(), lines=()):
    """boxes: [(box, label, (b, g, r))], lines: Text oben links."""
    out = img.copy()
    for box, label, color in boxes:
        x1, y1, x2, y2 = box_to_pixels(box, (out.shape[1], out.shape[0]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for i, text in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def render_map_view(occ, sm, rover=None, highlight=(), done=(), marks=()):
    """Draufsicht der Karte mit Raumnamen und Rover-Pfeil (für den VLM-Prompt).

    highlight: Räume, die in Frage kommen; done: schon durchsuchte Räume;
    marks: [(x, y, label)] z. B. gefundene Objekte.
    """
    img = cv2.cvtColor(occ.image, cv2.COLOR_GRAY2BGR)
    img = np.where(img > 127, 255, 40).astype(np.uint8)
    scale = 2
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        i, j = occ.to_px(x, y)
        return i * scale, j * scale

    overlay = img.copy()
    for name, room in sm['rooms'].items():
        x0, y0, x1, y1 = room['rect']
        color = (200, 200, 200) if name in done else (
            (160, 230, 160) if name in highlight else (235, 225, 200))
        cv2.rectangle(overlay, px(x0, y1), px(x1, y0), color, -1)
    img = np.where(img == 255, overlay, img).astype(np.uint8)
    for name, room in sm['rooms'].items():
        x0, y0, x1, y1 = room['rect']
        u, v = px((x0 + x1) / 2, (y0 + y1) / 2)
        size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(img, name, (u - size[0] // 2, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 40, 0), 2)
    for x, y, label in marks:
        cv2.circle(img, px(x, y), 7, (255, 80, 0), -1)
        cv2.putText(img, label, px(x + 0.2, y + 0.2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2)
    if rover is not None:
        x, y, yaw = rover
        a = px(x, y)
        b = px(x + 0.9 * math.cos(yaw), y + 0.9 * math.sin(yaw))
        cv2.arrowedLine(img, a, b, (0, 0, 230), 4, tipLength=0.45)
    i0, j0 = occ.to_px(-10.3, 7.3)
    i1, j1 = occ.to_px(10.3, -7.3)
    return img[j0 * scale:j1 * scale, i0 * scale:i1 * scale]
