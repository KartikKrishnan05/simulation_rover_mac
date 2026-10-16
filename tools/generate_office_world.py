#!/usr/bin/env python3
"""Erzeugt worlds/office.sdf – eine Büroumgebung für den Rover.

Grundriss (20 m x 14 m, Blick von oben, +x nach rechts, +y nach oben):

    y=7  +-----------+-----------+-------------------+
         | Büro A    | Büro B    | Küche             |
    y=3  +--  -------+--  -------+---    ------------+
         |                                           |
         |  Großraum mit Schreibtischinseln  Lounge  |
         |                         +-----------------+ y=-1 (Glaswand)
         |                         | Meetingraum     |
    y=-7 +-------------------------+-----------------+
       x=-10                      x=3              x=10

Zusätzlich wird aus denselben Kollisionsgeometrien die Belegungskarte für
Nav2 erzeugt (ros/maps/office.pgm + office.yaml).

Layout anpassen: Werte unten ändern und neu erzeugen mit
    python3 tools/generate_office_world.py
"""
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "worlds" / "office.sdf"
MAP_DIR = ROOT / "ros" / "maps"

# Karte: Auflösung und Ausschnitt (Weltkoordinaten in m)
MAP_RES = 0.05
MAP_X0, MAP_Y0 = -10.5, -7.5
MAP_X1, MAP_Y1 = 10.5, 7.5
# Nur Hindernisse in diesem Höhenbereich landen in der Karte
# (Rover inkl. Kameramast ist ca. 0,8 m hoch).
MAP_Z_MIN, MAP_Z_MAX = 0.03, 0.85

# Kollisionsprimitive des gerade entstehenden Modells (Modell-Koordinaten)
_pending = []
# Alle Hindernisse in Weltkoordinaten: ("box", x, y, yaw, sx, sy) / ("circle", x, y, r)
OBSTACLES = []

WALL_H = 2.5
WALL_T = 0.15

WALL_COLOR = "0.92 0.91 0.88"
GLASS_COLOR = "0.7 0.85 0.95"


def material(rgb, alpha_spec="0.1 0.1 0.1"):
    r, g, b = (float(v) for v in rgb.split())
    amb = f"{r * 0.8:.3f} {g * 0.8:.3f} {b * 0.8:.3f} 1"
    return (f"<material><ambient>{amb}</ambient><diffuse>{rgb} 1</diffuse>"
            f"<specular>{alpha_spec} 1</specular></material>")


def box_geom(sx, sy, sz):
    return f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"


def cyl_geom(r, h):
    return f"<geometry><cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>"


def part(name, pose, geom, rgb, collide=True, transparency=None):
    """Ein Visual (+ optional Collision) innerhalb eines Links."""
    p = " ".join(f"{v:.3f}" for v in pose)
    s = ""
    if collide:
        _pending.append((pose, geom))
        s += f'<collision name="{name}_col"><pose>{p}</pose>{geom}</collision>\n'
    t = f"<transparency>{transparency}</transparency>" if transparency else ""
    s += f'<visual name="{name}"><pose>{p}</pose>{geom}{material(rgb)}{t}</visual>\n'
    return s


def _commit_obstacles(x, y, yaw):
    """Kollisionsprimitive des Modells in Weltkoordinaten übernehmen."""
    c, s_ = math.cos(yaw), math.sin(yaw)
    for (px, py, pz, *_), geom in _pending:
        wx, wy = x + c * px - s_ * py, y + s_ * px + c * py
        if m := re.search(r"<box><size>(\S+) (\S+) (\S+)</size>", geom):
            sx, sy, sz = map(float, m.groups())
            if pz + sz / 2 > MAP_Z_MIN and pz - sz / 2 < MAP_Z_MAX:
                OBSTACLES.append(("box", wx, wy, yaw, sx, sy))
        elif m := re.search(r"<cylinder><radius>(\S+)</radius><length>(\S+)</length>", geom):
            r, h = map(float, m.groups())
            if pz + h / 2 > MAP_Z_MIN and pz - h / 2 < MAP_Z_MAX:
                OBSTACLES.append(("circle", wx, wy, r))
        elif m := re.search(r"<sphere><radius>(\S+)</radius>", geom):
            r = float(m.group(1))
            if pz + r > MAP_Z_MIN and pz - r < MAP_Z_MAX:
                OBSTACLES.append(("circle", wx, wy, r))
    _pending.clear()


def static_model(name, x, y, yaw, body):
    _commit_obstacles(x, y, yaw)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose>
      <link name="link">
{body}      </link>
    </model>"""


def dynamic_model(name, x, y, yaw, mass, ixx, iyy, izz, body, com_z=0.0):
    # Bewegliche Objekte kommen mit ihrer Startposition in die Karte.
    _commit_obstacles(x, y, yaw)
    return f"""
    <model name="{name}">
      <pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose>
      <link name="link">
        <inertial>
          <pose>0 0 {com_z:.3f} 0 0 0</pose>
          <mass>{mass}</mass>
          <inertia><ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{iyy}</iyy><iyz>0</iyz><izz>{izz}</izz></inertia>
        </inertial>
{body}      </link>
    </model>"""


# ---------------------------------------------------------------- Wände
def wall_segments_x(y, x1, x2, gaps=()):
    """Wand parallel zur x-Achse von x1 bis x2 bei y; gaps = [(mitte, breite)]."""
    cuts = sorted((c - w / 2, c + w / 2) for c, w in gaps)
    segs, start = [], x1
    for a, b in cuts:
        segs.append((start, a))
        start = b
    segs.append((start, x2))
    return [((a + b) / 2, y, b - a, WALL_T) for a, b in segs if b - a > 1e-3]


def wall_segments_y(x, y1, y2, gaps=()):
    cuts = sorted((c - w / 2, c + w / 2) for c, w in gaps)
    segs, start = [], y1
    for a, b in cuts:
        segs.append((start, a))
        start = b
    segs.append((start, y2))
    return [(x, (a + b) / 2, WALL_T, b - a) for a, b in segs if b - a > 1e-3]


def walls():
    solid = []
    # Außenwände (leicht verlängert, damit die Ecken geschlossen sind)
    e = WALL_T / 2
    solid += wall_segments_x(-7, -10 - e, 10 + e)
    solid += wall_segments_x(7, -10 - e, 10 + e)
    solid += wall_segments_y(-10, -7, 7)
    solid += wall_segments_y(10, -7, 7)
    # Trennwand zu Büros/Küche mit drei (Doppel-)Türen. Breite Türen, weil der
    # Ackermann-Rover beim Einlenken mit der Front ausschwenkt.
    solid += wall_segments_x(3, -10, 10, gaps=[(-7.2, 2.0), (-1.0, 2.0), (4.2, 2.2)])
    solid += wall_segments_y(-4, 3, 7)
    solid += wall_segments_y(2, 3, 7)
    # Meetingraum: Westwand mit Tür
    solid += wall_segments_y(3, -7, -1, gaps=[(-2.1, 2.0)])

    body = ""
    for i, (x, y, sx, sy) in enumerate(solid):
        body += part(f"wall_{i}", (x, y, WALL_H / 2, 0, 0, 0), box_geom(sx, sy, WALL_H), WALL_COLOR)

    # Glaswand zwischen Lounge und Meetingraum (mit Brüstung und Rahmen)
    glass_y, gx1, gx2 = -1, 3, 10
    gcx, glen = (gx1 + gx2) / 2, gx2 - gx1
    body += part("glass_sill", (gcx, glass_y, 0.15, 0, 0, 0), box_geom(glen, WALL_T, 0.3), "0.35 0.35 0.38")
    body += part("glass", (gcx, glass_y, 1.35, 0, 0, 0), box_geom(glen, 0.03, 2.1), GLASS_COLOR,
                 transparency=0.75)
    body += part("glass_top", (gcx, glass_y, 2.45, 0, 0, 0), box_geom(glen, WALL_T, 0.1), "0.35 0.35 0.38")
    return static_model("office_walls", 0, 0, 0, body)


# ---------------------------------------------------------------- Böden (nur Optik)
def floors():
    areas = [
        # name, cx, cy, sx, sy, Farbe
        ("open_space", -3.5, -2.0, 13.0, 10.0, "0.42 0.46 0.52"),   # Teppich blau-grau
        ("lounge", 6.5, 1.0, 7.0, 4.0, "0.55 0.5 0.45"),
        ("meeting", 6.5, -4.0, 7.0, 6.0, "0.3 0.32 0.38"),
        ("office_a", -7.0, 5.0, 6.0, 4.0, "0.62 0.45 0.3"),         # Holz
        ("office_b", -1.0, 5.0, 6.0, 4.0, "0.62 0.45 0.3"),
        ("kitchen", 6.0, 5.0, 8.0, 4.0, "0.85 0.85 0.82"),          # Fliesen
    ]
    body = ""
    for name, cx, cy, sx, sy, rgb in areas:
        body += part(name, (cx, cy, 0.002, 0, 0, 0), box_geom(sx, sy, 0.004), rgb, collide=False)
    return static_model("floors", 0, 0, 0, body)


# ---------------------------------------------------------------- Möbel
counter = {}


def uid(kind):
    counter[kind] = counter.get(kind, 0) + 1
    return f"{kind}_{counter[kind]}"


def desk(x, y, yaw, with_monitor=True):
    """Schreibtisch 1,4 x 0,7 m. Der Sitzplatz liegt auf der lokalen -y-Seite."""
    top = "0.78 0.72 0.62"
    leg = "0.25 0.25 0.27"
    b = part("top", (0, 0, 0.73, 0, 0, 0), box_geom(1.4, 0.7, 0.04), top)
    for i, (lx, ly) in enumerate([(0.66, 0.31), (-0.66, 0.31), (0.66, -0.31), (-0.66, -0.31)]):
        b += part(f"leg{i}", (lx, ly, 0.355, 0, 0, 0), box_geom(0.05, 0.05, 0.71), leg)
    b += part("panel", (0, 0.32, 0.45, 0, 0, 0), box_geom(1.3, 0.02, 0.45), top)
    if with_monitor:
        b += part("monitor", (0, 0.18, 1.0, 0, 0, 0), box_geom(0.6, 0.03, 0.36), "0.08 0.08 0.1")
        b += part("monitor_stand", (0, 0.2, 0.8, 0, 0, 0), box_geom(0.06, 0.06, 0.12), "0.2 0.2 0.2")
        b += part("keyboard", (0, -0.1, 0.76, 0, 0, 0), box_geom(0.45, 0.15, 0.02), "0.15 0.15 0.15")
    return static_model(uid("desk"), x, y, yaw, b)


def chair(x, y, yaw):
    """Bürostuhl (dynamisch, kann vom Rover verschoben werden). Lehne auf lokaler -y-Seite."""
    c = "0.15 0.15 0.18"
    b = part("base", (0, 0, 0.03, 0, 0, 0), cyl_geom(0.3, 0.06), "0.3 0.3 0.3")
    b += part("post", (0, 0, 0.26, 0, 0, 0), cyl_geom(0.03, 0.4), "0.4 0.4 0.4")
    b += part("seat", (0, 0, 0.48, 0, 0, 0), box_geom(0.48, 0.46, 0.07), c)
    b += part("back", (0, -0.22, 0.8, 0, 0, 0), box_geom(0.46, 0.05, 0.55), c)
    return dynamic_model(uid("chair"), x, y, yaw, 8.0, 0.3, 0.3, 0.2, b, com_z=0.35)


def chair_at_desk(dx, dy, dyaw):
    # Stuhl 0,75 m vor dem Tisch (lokal -y), Blick zum Tisch
    cx = dx + 0.75 * math.sin(dyaw)
    cy = dy - 0.75 * math.cos(dyaw)
    return chair(cx, cy, dyaw)


def desk_island(cx, cy):
    """Vier Arbeitsplätze, je zwei gegenüber."""
    s = ""
    for dx in (-0.7, 0.7):
        s += desk(cx + dx, cy - 0.35, 0.0) + chair_at_desk(cx + dx, cy - 0.35, 0.0)
        s += desk(cx + dx, cy + 0.35, math.pi) + chair_at_desk(cx + dx, cy + 0.35, math.pi)
    return s


def bookshelf(x, y, yaw):
    """Regal 0,9 x 0,35 x 1,8 m, Rückwand auf lokaler +y-Seite."""
    wood = "0.55 0.4 0.28"
    b = part("body_back", (0, 0.16, 0.9, 0, 0, 0), box_geom(0.9, 0.03, 1.8), wood)
    b += part("side_l", (-0.435, 0, 0.9, 0, 0, 0), box_geom(0.03, 0.35, 1.8), wood)
    b += part("side_r", (0.435, 0, 0.9, 0, 0, 0), box_geom(0.03, 0.35, 1.8), wood)
    colors = ["0.7 0.2 0.2", "0.2 0.4 0.7", "0.2 0.6 0.3", "0.8 0.7 0.2"]
    for i, z in enumerate([0.02, 0.45, 0.9, 1.35, 1.78]):
        b += part(f"shelf{i}", (0, 0, z, 0, 0, 0), box_geom(0.87, 0.33, 0.03), wood)
        if i < 4:
            b += part(f"books{i}", (-0.1 + 0.05 * i, 0.02, z + 0.15, 0, 0, 0),
                      box_geom(0.5, 0.22, 0.26), colors[i], collide=False)
    return static_model(uid("bookshelf"), x, y, yaw, b)


def plant(x, y, size=1.0):
    b = part("pot", (0, 0, 0.2 * size, 0, 0, 0), cyl_geom(0.18 * size, 0.4 * size), "0.75 0.4 0.25")
    b += part("stem", (0, 0, 0.55 * size, 0, 0, 0), cyl_geom(0.02, 0.4 * size), "0.35 0.25 0.15", collide=False)
    b += part("leaves", (0, 0, 0.9 * size, 0, 0, 0),
              f"<geometry><sphere><radius>{0.35 * size:.3f}</radius></sphere></geometry>", "0.2 0.55 0.2")
    return static_model(uid("plant"), x, y, 0, b)


def table(x, y, yaw, sx, sy, h=0.74, top="0.85 0.85 0.83"):
    b = part("top", (0, 0, h - 0.02, 0, 0, 0), box_geom(sx, sy, 0.04), top)
    lx, ly = sx / 2 - 0.08, sy / 2 - 0.08
    for i, (px, py) in enumerate([(lx, ly), (-lx, ly), (lx, -ly), (-lx, -ly)]):
        b += part(f"leg{i}", (px, py, (h - 0.04) / 2, 0, 0, 0), box_geom(0.05, 0.05, h - 0.04), "0.3 0.3 0.3")
    return static_model(uid("table"), x, y, yaw, b)


def round_table(x, y, r=0.5, h=0.75):
    b = part("top", (0, 0, h - 0.02, 0, 0, 0), cyl_geom(r, 0.04), "0.95 0.95 0.95")
    b += part("post", (0, 0, (h - 0.04) / 2, 0, 0, 0), cyl_geom(0.04, h - 0.04), "0.4 0.4 0.4")
    b += part("foot", (0, 0, 0.015, 0, 0, 0), cyl_geom(0.25, 0.03), "0.4 0.4 0.4")
    return static_model(uid("round_table"), x, y, 0, b)


def sofa(x, y, yaw):
    """Sofa 2,0 x 0,85 m, Rückenlehne auf lokaler +y-Seite."""
    c = "0.3 0.45 0.6"
    b = part("seat", (0, -0.05, 0.22, 0, 0, 0), box_geom(2.0, 0.75, 0.44), c)
    b += part("back", (0, 0.33, 0.6, 0, 0, 0), box_geom(2.0, 0.2, 0.5), c)
    b += part("arm_l", (-0.95, 0, 0.55, 0, 0, 0), box_geom(0.1, 0.85, 0.3), c)
    b += part("arm_r", (0.95, 0, 0.55, 0, 0, 0), box_geom(0.1, 0.85, 0.3), c)
    return static_model(uid("sofa"), x, y, yaw, b)


def block(name, x, y, yaw, sx, sy, sz, rgb, z=0.0):
    """Quader; z = Höhe der Unterkante über dem Boden."""
    return static_model(uid(name), x, y, yaw,
                        part("body", (0, 0, z + sz / 2, 0, 0, 0), box_geom(sx, sy, sz), rgb))


def cardboard_box(x, y, yaw, s=0.4):
    m = 2.0
    i = m * (2 * s * s) / 12
    b = part("body", (0, 0, s / 2, 0, 0, 0), box_geom(s, s, s), "0.7 0.55 0.35")
    return dynamic_model(uid("box"), x, y, yaw, m, i, i, i, b, com_z=s / 2)


def furniture():
    s = ""
    # Großraum: drei Schreibtischinseln. Zwischen den Inseln und zu den Türen
    # bleibt genug Platz zum Rangieren (Wendekreisradius des Rovers ≈ 1 m).
    for cx, cy in [(-7.0, -4.9), (-2.5, -4.9), (-7.0, -0.5)]:
        s += desk_island(cx, cy)
    # Regale an der Westwand (Rückwand zur Wand: yaw -90°)
    for y in (-6.0, -2.0, 1.8):
        s += bookshelf(-9.75, y, math.pi / 2)
    s += plant(-9.0, -6.5)
    s += plant(2.5, -6.5)
    s += plant(2.5, 2.5, 0.8)
    s += block("water_cooler", 0.5, 2.6, 0, 0.35, 0.35, 1.2, "0.85 0.9 0.95")
    s += block("printer", 1.8, -6.5, 0, 0.6, 0.5, 1.0, "0.82 0.82 0.8")
    # ein paar Kartons zum Wegschieben (abseits der Fahrwege)
    s += cardboard_box(0.6, -6.5, 0.3)
    s += cardboard_box(0.0, -6.55, 0.0, 0.5)
    s += cardboard_box(-9.5, 2.6, 0.7)

    # Lounge: Sitzecke im Osten, damit im Westteil Platz zum Wenden bleibt
    # (der Rover braucht für eine Kehrtwende ≈ 2,5 m)
    s += sofa(8.0, 2.4, 0.0)
    s += table(8.0, 1.55, 0, 1.2, 0.6, h=0.42, top="0.55 0.4 0.28")
    s += plant(9.5, 2.5)
    s += plant(9.6, -0.5, 0.9)

    # Meetingraum
    s += table(6.8, -4.0, 0, 3.0, 1.2)
    for x in (5.8, 6.8, 7.8):
        s += chair(x, -4.95, 0.0)
        s += chair(x, -3.05, math.pi)
    s += chair(4.75, -4.0, -math.pi / 2)
    s += chair(8.85, -4.0, math.pi / 2)
    s += block("screen", 9.88, -4.0, 0, 0.06, 1.6, 0.9, "0.05 0.05 0.07", z=1.0)
    s += plant(9.5, -6.5, 0.9)

    # Büro A
    s += desk(-7.0, 6.45, 0.0) + chair_at_desk(-7.0, 6.45, 0.0)
    s += bookshelf(-4.3, 5.5, -math.pi / 2)
    s += plant(-9.5, 6.5)
    # Büro B
    s += desk(-1.0, 6.45, 0.0) + chair_at_desk(-1.0, 6.45, 0.0)
    s += bookshelf(-3.7, 5.5, math.pi / 2)
    s += plant(1.5, 6.5)
    s += cardboard_box(1.2, 3.7, 0.4)

    # Küche
    s += block("counter", 5.5, 6.62, 0, 4.0, 0.6, 0.9, "0.9 0.9 0.88")
    s += block("fridge", 9.5, 6.55, 0, 0.7, 0.7, 1.85, "0.8 0.82 0.85")
    s += block("trash_bin", 2.4, 6.6, 0, 0.35, 0.35, 0.6, "0.3 0.3 0.3")
    # Tisch im hinteren Teil, damit der Rover hinter der Tür wenden kann
    s += round_table(8.3, 4.8)
    for a in (0.0, 2.1, 4.2):
        s += chair(8.3 + 0.85 * math.cos(a), 4.8 + 0.85 * math.sin(a), a - math.pi / 2)
    return s


def lights():
    s = ""
    for i, (x, y) in enumerate([(-6, -4), (-1, -4), (-6, 0.5), (-1, 0.5), (6.5, 1),
                                (6.5, -4), (-7, 5), (-1, 5), (6, 5)]):
        s += f"""
    <light type="point" name="ceiling_light_{i}">
      <pose>{x} {y} 2.4 0 0 0</pose>
      <diffuse>0.5 0.5 0.48 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation><range>8</range><constant>0.6</constant><linear>0.08</linear><quadratic>0.01</quadratic></attenuation>
      <cast_shadows>false</cast_shadows>
    </light>"""
    return s


# ---------------------------------------------------------------- Karte für Nav2
def _occupied(x, y, ob):
    if ob[0] == "circle":
        _, cx, cy, r = ob
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    _, cx, cy, yaw, sx, sy = ob
    dx, dy = x - cx, y - cy
    c, s_ = math.cos(yaw), math.sin(yaw)
    lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
    return abs(lx) <= sx / 2 and abs(ly) <= sy / 2


def write_map():
    w = round((MAP_X1 - MAP_X0) / MAP_RES)
    h = round((MAP_Y1 - MAP_Y0) / MAP_RES)
    grid = bytearray([254]) * (w * h)  # frei
    for ob in OBSTACLES:
        reach = max(ob[4], ob[5]) * 0.75 if ob[0] == "box" else ob[3]
        i0 = max(0, int((ob[1] - reach - MAP_X0) / MAP_RES))
        i1 = min(w - 1, int((ob[1] + reach - MAP_X0) / MAP_RES) + 1)
        j0 = max(0, int((ob[2] - reach - MAP_Y0) / MAP_RES))
        j1 = min(h - 1, int((ob[2] + reach - MAP_Y0) / MAP_RES) + 1)
        for j in range(j0, j1 + 1):
            y = MAP_Y0 + (j + 0.5) * MAP_RES
            for i in range(i0, i1 + 1):
                x = MAP_X0 + (i + 0.5) * MAP_RES
                if _occupied(x, y, ob):
                    grid[(h - 1 - j) * w + i] = 0  # PGM: erste Zeile = oben
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    (MAP_DIR / "office.pgm").write_bytes(f"P5\n{w} {h}\n255\n".encode() + bytes(grid))
    (MAP_DIR / "office.yaml").write_text(
        "# Erzeugt von tools/generate_office_world.py – nicht von Hand bearbeiten.\n"
        "image: office.pgm\n"
        "mode: trinary\n"
        f"resolution: {MAP_RES}\n"
        f"origin: [{MAP_X0}, {MAP_Y0}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n")


def main():
    template = (ROOT / "tools" / "office_template.sdf").read_text()
    sdf = (template
           .replace("<!--@WALLS@-->", walls() + floors())
           .replace("<!--@FURNITURE@-->", furniture())
           .replace("<!--@LIGHTS@-->", lights()))
    OUT.write_text(sdf)
    print(f"geschrieben: {OUT.relative_to(ROOT)}")
    write_map()
    print(f"geschrieben: {(MAP_DIR / 'office.pgm').relative_to(ROOT)} "
          f"({len(OBSTACLES)} Hindernisse)")


if __name__ == "__main__":
    main()
