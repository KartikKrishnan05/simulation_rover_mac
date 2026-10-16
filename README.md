# Rover-Simulation in Gazebo mit ROS 2 Navigation (macOS)

Ein Rover mit **Ackermann-Lenkung** fährt autonom durch eine Büroumgebung –
simuliert in **Gazebo Harmonic**, navigiert mit **ROS 2 Jazzy / Nav2**. Alles
läuft nativ auf macOS (Apple Silicon).

Das Büro (20 m × 14 m) besteht aus einem Großraum mit drei Schreibtischinseln,
einer Lounge, einem Meetingraum hinter einer Glaswand, zwei Einzelbüros und
einer Küche. Stühle und Kartons sind beweglich.

```
.
├── run.sh                         # Startet Gazebo (Server + GUI)
├── install_gazebo_mac.sh          # Installiert Gazebo Harmonic über Homebrew
├── pixi.toml                      # ROS 2 Jazzy + Nav2 (RoboStack) über pixi
├── models/rover/                  # Rover-Modell (SDF, Ackermann)
├── worlds/office.sdf              # Büro-Welt (generiert)
├── tools/
│   ├── generate_office_world.py   # Grundriss & Möbel → Welt + Nav2-Karte
│   └── office_template.sdf        # Physik, Plugins, GUI, Rover
├── ros/
│   ├── launch/navigation.launch.py  # Bridge, Lokalisierung, Nav2, RViz
│   ├── config/nav2_params.yaml      # Nav2 für Ackermann
│   ├── config/ekf.yaml              # Rad-Odometrie + Gyroskop (robot_localization)
│   ├── config/bridge.yaml           # Gazebo ↔ ROS-Topics
│   ├── config/cyclonedds.xml        # DDS nur über Loopback
│   ├── behavior_trees/              # BTs ohne „auf der Stelle drehen“
│   ├── maps/office.{pgm,yaml}       # Karte (aus der Welt generiert)
│   ├── urdf/rover.urdf              # für robot_state_publisher / RViz
│   └── scripts/
│       ├── send_goal.py             # Ziele / Rundfahrt per Nav2
│       └── scan_merger.py           # 2 Lidare → 360°-Scan für AMCL
└── scripts/
    ├── teleop.sh                  # Steuerung per Terminal (ohne ROS)
    └── drive_demo.sh              # kurze Demo-Fahrt (ohne ROS)
```

## Der Rover

| | |
|---|---|
| Lenkung | Ackermann: Vorderräder an Achsschenkeln, kurveninneres Rad schlägt stärker ein |
| Antrieb | Hinterräder (Gazebo-System `AckermannSteering`) |
| Maße | 1,0 m × 0,76 m, Radstand 0,56 m, Spurweite 0,62 m |
| Wendekreisradius | ≈ 0,87 m (Lenkwinkel max. 0,7 rad) – **kein Drehen auf der Stelle** |
| Sensoren | Lidar vorne + hinten (je 185°), Kamera, IMU |
| Odometrie | EKF aus Rad-Odometrie (Geschwindigkeit) + Gyroskop (Gierrate), Bezugspunkt Mitte Hinterachse (`base_footprint`) |

In Gazebo nachgemessen: Ein Drehbefehl im Stand bewegt den Rover nicht, der
gefahrene Kreis hat 0,91 m Radius, und die Lenkwinkel innen/außen entsprechen der
Ackermann-Geometrie.

## 1. Installation

Voraussetzungen: [Homebrew](https://brew.sh) und die Xcode Command Line Tools.

```bash
./install_gazebo_mac.sh
```

```bash
brew install pixi
```

```bash
pixi install
```

`install_gazebo_mac.sh` installiert Gazebo Harmonic (`brew tap osrf/simulation`,
`brew trust osrf/simulation`, `brew install gz-harmonic`). `pixi install` holt
ROS 2 Jazzy mit Nav2, slam_toolbox, robot_localization, ros_gz_bridge und RViz von
[RoboStack](https://robostack.github.io) in den Ordner `.pixi/` – ROS wird nicht
systemweit installiert.

## 2. Starten

Terminal 1 – Gazebo:

```bash
./run.sh
```

Terminal 2 – Nav2 und RViz:

```bash
pixi run nav
```

Nach ca. 15 s ist Nav2 bereit. In RViz mit **„Nav2 Goal“** ein Ziel auf der
Karte setzen – der Rover plant einen fahrbaren Pfad (inkl. Rangieren rückwärts)
und fährt los. Alternativ per Skript:

```bash
pixi run goal kitchen
```

```bash
pixi run goal tour
```

Bekannte Orte: `start`, `desks`, `office_a`, `office_b`, `kitchen`, `lounge`,
`meeting` (`pixi run goal -- --list`), oder eine Pose `x y yaw_grad`.

**Karte selbst erstellen** (SLAM statt fertiger Karte):

```bash
pixi run slam
```

## 3. Manuell steuern

- **Gazebo-GUI**: Panel „Rover-Steuerung“ oder in die 3D-Ansicht klicken und
  Pfeiltasten / W A S D (links/rechts fahren eine Kurve), Leertaste = Stopp.
- **Terminal ohne ROS**: `./scripts/teleop.sh`
- **Terminal mit ROS**: `pixi run teleop`

Nav2 und manuelle Befehle nutzen beide `/cmd_vel` – während einer Navigation
nicht gleichzeitig manuell steuern.

## Wie die Navigation auf Ackermann abgestimmt ist

| Baustein | Wahl | Warum |
|---|---|---|
| Globaler Planer | Smac **Hybrid-A\***, Reeds-Shepp, `minimum_turning_radius: 0.95` | erzeugt nur Pfade, die das Fahrzeug fahren kann, inkl. Rückwärtsrangieren |
| Regler | **Regulated Pure Pursuit**, `allow_reversing`, ohne „rotate to heading“ | folgt dem fahrbaren Pfad vor- und rückwärts (MPPI mit Ackermann-Modell pendelte in den engen Räumen) |
| Behavior Tree | Standard-BT ohne `Spin` | ein Auto kann nicht auf der Stelle drehen |
| Footprint | Rechteck um die Hinterachse | `base_footprint` liegt dort, wo die Odometrie ansetzt |
| Odometrie | EKF: Räder + Gyroskop | die Rad-Odometrie allein driftet beim Rangieren (Schlupf) |
| Lokalisierung | AMCL mit 360°-Scan aus beiden Lidaren | ein einzelner 185°-Scan war in Türen und Gängen mehrdeutig |
| Costmaps | Karte + beide Lidare, Planer mit 10 cm Polster | die Lidare (0,27 m) sehen keine Tischplatten; die Front schwenkt in Kurven aus |

## Teststand und bekannte Grenzen

- Ackermann-Physik in Gazebo nachgemessen (siehe oben).
- Nav2 startet in ~15 s vollständig; AMCL liegt mit dem 360°-Scan typischerweise
  2–10 cm neben der Ground Truth.
- Die komplette Rundfahrt (`pixi run goal tour`, 6 Ziele) lief zweimal
  hintereinander ohne Fehler durch, je ca. 3–4 Minuten (Mac am Netzteil).
- Genauigkeit: AMCL liegt 5–25 cm neben der Ground Truth, der Rover hält am
  Ziel 0,3–0,7 m vor dem Sollpunkt (Zieltoleranz 0,35 m plus
  Lokalisierungsfehler).
- Im Meetingraum ist kein Platz zum Wenden (1,7 m zwischen Stühlen und
  Glaswand); das Ziel `meeting` liegt deshalb direkt hinter der Tür.
- RViz meldet beim Start unter macOS eine Shader-Warnung (`indexed_8bit_image`)
  und läuft weiter; die Darstellung von Karte/Costmaps ist nicht geprüft.
- Die Simulation reagiert empfindlich auf eine ausgelastete Maschine sowie auf
  Ruhezustand und Stromsparmodus: Dann hängt die Sim-Uhr kurz und Nav2 bricht
  Ziele ab.

## Topics (ROS 2)

| Topic | Typ | Beschreibung |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Fahrbefehl (linear.x, angular.z → Lenkwinkel) |
| `/odom` | `nav_msgs/Odometry` | Rad-Odometrie |
| `/scan`, `/scan_rear` | `sensor_msgs/LaserScan` | Lidar vorne / hinten |
| `/scan_merged` | `sensor_msgs/LaserScan` | 360°-Scan (für AMCL/SLAM) |
| `/camera/image_raw` | `sensor_msgs/Image` | Kamera 640×480 |
| `/imu` | `sensor_msgs/Imu` | IMU |
| `/joint_states` | `sensor_msgs/JointState` | Lenkung und Räder |

## Büro anpassen

Wände, Türen und Möbel stehen in `tools/generate_office_world.py`. Das Skript
erzeugt die Gazebo-Welt **und** die passende Nav2-Karte aus denselben
Kollisionsgeometrien:

```bash
python3 tools/generate_office_world.py
```

Türen sind 2,0–2,2 m breit: Ein Ackermann-Fahrzeug schwenkt beim Einlenken mit
der Front aus und braucht mehr Platz als ein Differentialantrieb.

## Problembehebung

- **GUI bleibt leer**: Server noch nicht fertig – GUI mit `gz sim -g` erneut
  starten. `GZ_IP=127.0.0.1` ist in den Skripten und in der pixi-Umgebung gesetzt.
- **Nav2 findet keine Daten / Rover bewegt sich nicht**: Läuft Gazebo
  (`./run.sh`) und ist die Simulation nicht pausiert?
- **Rechner ausgelastet**: Gazebo rendert Kamera und zwei Lidare; zusammen mit
  Nav2 braucht das spürbar CPU. Andere rechenintensive Programme schließen.
  Den Mac nicht in den Ruhezustand gehen lassen, während die Simulation läuft.
- **Alte Prozesse**: `pkill -f "gz sim"` und Nav2 im zweiten Terminal mit
  Ctrl+C beenden.
