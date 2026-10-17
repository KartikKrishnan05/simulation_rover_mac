# Rover-Simulation in Gazebo mit ROS 2 Navigation (macOS)

Ein Rover mit **Ackermann-Lenkung** fährt autonom durch eine Büroumgebung –
simuliert in **Gazebo Harmonic**, navigiert mit **ROS 2 Jazzy / Nav2**. Alles
läuft nativ auf macOS (Apple Silicon).

Das Büro (20 m × 14 m) besteht aus einem Großraum mit drei Schreibtischinseln,
einer Lounge, einem Meetingraum hinter einer Glaswand, zwei Einzelbüros und
einer Küche. Stühle und Kartons sind beweglich.

Dazu kommt ein **VLM-Agent** mit einem lokalen Vision-Language-Model
(Qwen3-VL 4B über Ollama): Er versteht Befehle wie „go to the meeting room“,
„find the blue bag“ oder „follow the person“, entscheidet anhand von Karte,
Kamera und Lidar und lässt Nav2 fahren. Jede Ausführung wird als Episode für
ein späteres VLA (Vision-Language-Action-Modell) aufgezeichnet – siehe
[Abschnitt 4](#4-vlm-agent-sprachbefehle).

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
│   ├── rviz/rover_vlm.rviz          # Nav2-Ansicht + Kamerabild/Marker des Agenten
│   ├── config/semantic_map.yaml     # Räume, Aussichtspunkte, Taschen-Orte, Weg der Person
│   ├── scripts/
│   │   ├── send_goal.py             # Ziele / Rundfahrt per Nav2
│   │   └── scan_merger.py           # 2 Lidare → 360°-Scan für AMCL
│   └── vlm/                         # VLM-Agent (siehe Abschnitt 4)
│       ├── agent.py                 # Sprachbefehl → goto / find / follow
│       ├── vlm_client.py            # Ollama-Client (lokales Modell)
│       ├── perception.py            # Kamera + Tiefe → Kartenpunkte, Orakel
│       ├── tracker.py               # Personen-Tracker auf den Lidaren
│       ├── person_walker.py         # lässt die Person ihren Weg ablaufen
│       ├── episode_recorder.py      # Episoden für ein späteres VLA
│       ├── collect.py               # Episoden sammeln + auswerten
│       ├── to_lerobot.py            # Episoden → LeRobot-Dataset
│       └── common.py                # Karte, Sim-Steuerung, Nav2-Client
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
| Sensoren | Lidar vorne + hinten (je 185°), Kamera 640×480, Tiefenkamera 320×240 (gleiches Sichtfeld), IMU |
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

Für den VLM-Agenten zusätzlich [Ollama](https://ollama.com) und das Modell
(≈ 3,3 GB):

```bash
brew install ollama
```

```bash
pixi run vlm-model
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

## 4. VLM-Agent (Sprachbefehle)

Gazebo und Nav2 wie oben starten (`./run.sh`, `pixi run nav` – startet auch
die Person und den Episoden-Recorder), Ollama muss laufen (`ollama serve` oder
die Ollama-App). Dann in Terminal 3:

```bash
pixi run agent "go to the meeting room"
```

```bash
pixi run agent "find the blue bag"
```

```bash
pixi run person "data: reset"
```

```bash
pixi run agent --start-person "follow the person"
```

Ohne Befehl startet ein interaktiver Modus (`pixi run agent`). `--reset`
setzt den Rover vorher an die Startpose (Gazebo + AMCL) und schickt die
Person an ihren Parkplatz (`pixi run person "data: park"` geht auch einzeln) –
nach dem Folgen steht sie sonst am Ende ihres Wegs vor der Startpose. In RViz zeigt das
Panel **VLM Kamera** das Kamerabild mit der Bounding-Box des Modells, die
Karte zeigt gefundene Objekte, die verfolgte Person und das aktuelle Ziel.

### Wie der Agent entscheidet

Das VLM trifft die Entscheidungen, Nav2 fährt. Ein 4B-Modell ist nicht
zuverlässig genug, um frei Werkzeuge aufzurufen – deshalb sind die Abläufe
fest und das Modell beantwortet an jeder Stelle eine kleine, klare Frage:

| Schritt | Eingabe an das VLM | Antwort |
|---|---|---|
| Anweisung verstehen | Text + Raumliste aus `semantic_map.yaml` | `{"task": "goto"/"find"/"follow", "room", "target"}` |
| Räume ordnen (find) | Draufsicht der Karte mit Raumnamen und Rover + Raumbeschreibungen | Räume, wahrscheinlichster zuerst |
| Objekt suchen | Kamerabild | `{"found": true, "bbox_2d": […]}` |
| Gegenprobe | vergrößerter Bildausschnitt | yes / no – bestätigt ab P(yes) ≥ 0,2 |
| Raum prüfen (goto) | Kamerabild + Raumbeschreibung | yes / no |

Aus der Bounding-Box wird mit der Tiefenkamera und TF ein Punkt auf der Karte
(Abweichung zur Ground Truth gemessen 0,1–0,2 m – die Kamera sieht die
Vorderseite). Unplausible Punkte (unter dem Boden, in Wänden/Regalen) werden
verworfen.

Gesucht wird nicht nur an den Aussichtspunkten: **während der Fahrt** gibt
ein Hintergrund-Thread laufend Kamerabilder an das VLM (jedes Bild mit der
Kamerapose zum Aufnahmezeitpunkt). Das gleicht aus, dass ein Ackermann-Rover
am Ziel bis zu 34° schief stehen darf (`yaw_goal_tolerance`) und der Blick
dann am Objekt vorbeigeht. Nach dem Anfahren muss eine **zweite Erkennung aus
der Nähe** an derselben Stelle (< 1 m) liegen – zwei Blickwinkel, die sich
einig sind; sonst gilt der Treffer als Fehlalarm und die Suche geht weiter.

**Folgen:** Das VLM findet die Person nur am Anfang (und wenn die Spur
verloren geht). Dazwischen verfolgt `tracker.py` die Beine im Lidar (10 Hz),
der Agent ersetzt laufend das Nav2-Ziel 1,8 m hinter der Person und hält bei
weniger als 2 m an. Die Person (`person_walker.py`) läuft auf Kommando einen
festen Weg aus `semantic_map.yaml` mit 0,3 m/s (Großraum → Lounge, Kehre,
zurück).

**Nav2-Hänger:** Bleibt der Rover an einer Türkante hängen („Start occupied“),
setzt er sich 0,4 m vor oder zurück frei (Richtung mit mehr Platz) und
versucht es erneut. Den Meetingraum fährt er über eine Zwischenpose an
(`via` bei (-0,5, -2,1), ein durchgehender Pfad per NavigateThroughPoses):
Der Rover wendet im freien Großraum und kommt gerade durch die Tür. Die
via-Pose muss mehr als 0,7 m von der Startpose entfernt liegen – näher
verwirft der BT (`RemovePassedGoals`) sie sofort als „schon erreicht“.

### Lokales Modell auf 16 GB (gemessen auf einem M2)

| | |
|---|---|
| Modell | `qwen3-vl:4b-instruct` (3,3 GB, ≈ 4,8 GB im Speicher) – Gazebo-GUI, RViz und Nav2 laufen daneben |
| Neues Bild | ≈ 6–8 s (Ollama skaliert jedes Bild auf ≈ 1100 Token, unabhängig von der Auflösung) |
| Nur Text | ≈ 1 s (Anweisung verstehen, kurze Antworten) |
| Simulation | pausiert, solange das VLM rechnet und der Rover steht (`--no-pause` schaltet ab) |

Zwei Dinge haben sich als entscheidend erwiesen: **kurze Antworten**
(≈ 15 Token statt Listen mit hunderten) und **kein JSON-Schema über Ollamas
`format`** – die Grammatik läuft auf der CPU und war neben Gazebo 10–100×
langsamer. Gegenproben im selben Chat (Prompt-Cache, < 1 s) taugen beim
4B-Modell nicht, es bestätigt dann nur seine erste Antwort; die Gegenprobe
läuft deshalb mit einem eigenen Bildausschnitt. Entschieden wird über die
Token-Wahrscheinlichkeit (Ollama `logprobs`): Fehlalarme wie blaue Bücher im
Regal liegen bei P(yes) < 1 %, echte Treffer schwanken bei knappen Ansichten
um 50 % – die Schwelle 0,2 fängt sie auf. Anderes Modell:
`pixi run agent --model qwen3-vl:8b-instruct …` oder `VLM_MODEL=…`.

### Daten für ein späteres VLA

Jeder Agentenlauf wird als Episode gespeichert (`data/episodes/<id>/`, 10 Hz
Sim-Zeit, ≈ 0,6 MB/s):

| Datei | Inhalt |
|---|---|
| `meta.json` | Anweisung, Aufgabe, Erfolg (Agent und Ground Truth), Modell, VLM-Aufrufe |
| `steps.npz` | `state` (AMCL x, y, yaw), `velocity`, `steering`, **`action`** (befohlenes v, ω = Expertenlabel), `subgoal` (Nav2-Ziel), `ground_truth`, `scan` (720 Strahlen) |
| `rgb/*.jpg`, `depth/*.png` | Kamerabilder, Tiefe in mm |
| `events.jsonl` | jede VLM-Anfrage/-Antwort und Agenten-Entscheidung mit Schrittindex |

**Export nach LeRobot**: `to_lerobot.py` schreibt die Episoden als
[LeRobot](https://github.com/huggingface/lerobot)-Dataset (Format v3, 10 fps)
nach `data/lerobot/rover_office/`. Es läuft in einer eigenen pixi-Umgebung
(`lerobot`, ≈ 1,4 GB mit torch), damit die ROS-Umgebung schlank bleibt; der
erste Aufruf installiert sie:

```bash
pixi run -e lerobot to-lerobot
```

| LeRobot | Quelle |
|---|---|
| `observation.images.front` | `rgb/*.jpg` als Video (AV1) |
| `observation.images.depth` | `depth/*.png`, nur mit `--depth` (verlustfreies 12-Bit-HEVC, log-quantisiert bis 12 m) |
| `observation.state` | x, y, yaw (AMCL), v, ω (Odometrie), Lenkwinkel |
| `action` | befohlenes v, ω |
| `task` | `instruction` aus `meta.json` |

Standardmäßig landen nur erfolgreiche Episoden im Dataset (`gt_success`, sonst
`success`); `--all` nimmt alle. Woher jede Episode stammt, steht in
`meta/rover_episodes.jsonl`. Zum Abschluss lädt das Skript das Dataset mit
`LeRobotDataset` und prüft Frames und fps. Weitere Optionen: `--overwrite`,
`--vcodec h264_videotoolbox` (Hardware-Encoder), `--out`, `--repo-id`.
Beim Laden `video_backend="pyav"` angeben – torchcodec findet unter macOS ohne
systemweites FFmpeg seine Bibliotheken nicht.

Viele Episoden entstehen schneller mit dem **Orakel**: Es ersetzt das VLM
durch Ground Truth aus Gazebo (Objekt sichtbar = im Bild und nicht verdeckt)
und sucht Räume nach Entfernung ab – gleicher Ablauf, gleiche Daten, ohne
Wartezeit auf das Modell:

```bash
pixi run collect find --episodes 30 --perception oracle
```

```bash
pixi run collect find --episodes 6
```

```bash
pixi run collect follow --episodes 1
```

`collect.py` versetzt pro Episode die Tasche (sechs Orte, einer pro Raum),
setzt Rover und Person zurück, formuliert die Anweisung zufällig und prüft
den Erfolg mit Ground Truth. Ergebnisse: `data/episodes/results.jsonl`.

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

**VLM-Agent** (M2, 16 GB, Gazebo-GUI und RViz offen, `qwen3-vl:4b-instruct`,
Erfolg jeweils mit Ground Truth geprüft):

| Szenario | Ergebnis |
|---|---|
| find (`collect find`, VLM) | 3/3 – Küche, Büro A, Großraum (dazu Lounge einzeln); 4–45 VLM-Aufrufe, 1–7 min; Tasche 0,1–0,2 m genau lokalisiert |
| follow | 2/2 – ganzer Weg (~90 s) ohne Spurverlust, mittlerer Abstand 3,5 m, Tracker 0,1–0,6 m neben der Ground Truth |
| goto (`collect goto --episodes 6 --seed 3`) | 6/6 – alle Räume einmal (Lounge, Büro A, Meetingraum, Großraum, Büro B, Küche), ohne Freisetzen; 43–66 s pro Episode |
| goto Meetingraum von der Startpose (`agent --reset "go to the meeting room"`) | 5/5 ohne Freisetzen, 90–147 s; Sim lief dabei mit 0,7–0,9 × Echtzeit (Mac im Stromsparmodus) |

- **Meetingraum von der Startpose:** Der Rover steht mit Blick nach Westen
  und muss wenden. Früher lag die via-Pose (1,2, -2,1) nur 0,4 m vom Start
  entfernt; der BT verwarf sie sofort, und der Hybrid-A* plante das
  Wendemanöver direkt vor der Tür (Rückwärtszüge bis x ≈ 2,4, die Tür liegt
  bei x = 3). Der Rover verkeilte sich an der Türkante (etwa jeder vierte
  Versuch). Mit der via-Pose bei (-0,5, -2,1) wendet er im Großraum
  (Rückwärtszüge nur bei x < 0,8) und fährt gerade durch die Tür.
- **Pendeln an Wendepunkten:** Regulated Pure Pursuit wechselt an Cusps des
  Reeds-Shepp-Pfads manchmal mehrfach zwischen vor und zurück, bevor es
  weitergeht – beim Wenden für den Meetingraum 10–40 s, im freien Großraum
  und ohne Folgen. Zwei Gegenmittel wurden getestet und wieder verworfen:
  seltener neu planen (der BT hielt den Pfad, der Rover pendelte länger) und
  ein zweiter, reiner Vorwärtsplaner (Dubins) für NavigateThroughPoses
  (5/5, aber der Rover wich dennoch auf Rückwärtszüge aus, einmal vor der
  Tür, einmal schräg durch die Tür).
- Fahrzeiten und Timeouts des Agenten zählen in Sim-Zeit (`Navigator.wait`),
  `reset_rover` wartet, bis AMCL die Startpose übernommen hat. Vorher brach
  der Agent bei langsamer Sim (Stromsparmodus, ≈ 0,2 × Echtzeit) Fahrten
  nach 300 s Wanduhr ab, und direkt nach dem Zurücksetzen hielt er den Rover
  manchmal noch für im alten Raum (dann ohne via-Pose).
- Der Orakel-Modus wurde mit 2 Episoden geprüft (Ablauf und Daten wie mit
  VLM); viele Orakel-Episoden am Stück sind noch nicht gelaufen.
- Die neuen RViz-Panels (VLM Kamera, Marker) sind konfiguriert, aber nicht
  am Bildschirm geprüft.

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
| `/camera/depth` | `sensor_msgs/Image` | Tiefe 320×240 (32FC1, m) |
| `/rover/ground_truth`, `/person/ground_truth`, `/bag/ground_truth` | `tf2_msgs/TFMessage` | echte Posen aus Gazebo (Auswertung, Orakel) |
| `/person/command` | `std_msgs/String` | `start`, `reset`, `stop` für die Person |
| `/vlm/image`, `/vlm/markers`, `/vlm/map_view` | `Image`, `MarkerArray` | Anzeige des Agenten in RViz |
| `/vlm/events`, `/episode/control` | `std_msgs/String` (JSON) | Agent → Episoden-Recorder |
| `/world/office/control`, `/world/office/set_pose` | Services (`ros_gz_interfaces`) | Simulation pausieren, Objekte versetzen |

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
