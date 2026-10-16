#!/usr/bin/env bash
# Startet die Rover-Simulation unter macOS.
#
# Unter macOS kann Gazebo Server und GUI nicht in einem Prozess starten
# (`gz sim welt.sdf` funktioniert nicht), daher werden beide getrennt gestartet.
#
# Nutzung:
#   ./run.sh                    # Standardwelt mit GUI
#   ./run.sh pfad/zur/welt.sdf  # eigene Welt
#   HEADLESS=1 ./run.sh         # nur Server, ohne GUI
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD="${1:-$ROOT/worlds/office.sdf}"

if ! command -v gz >/dev/null 2>&1; then
  echo "Fehler: 'gz' nicht gefunden. Bitte zuerst ./install_gazebo_mac.sh ausführen." >&2
  exit 1
fi

export GZ_SIM_RESOURCE_PATH="$ROOT/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
# Gazebo-Transport auf Loopback beschränken: vermeidet Discovery-Probleme
# mit der macOS-Firewall bzw. wechselnden Netzwerken.
export GZ_IP="${GZ_IP:-127.0.0.1}"

echo "Starte Gazebo-Server mit $WORLD ..."
gz sim -s -r -v 3 "$WORLD" &
SERVER_PID=$!

cleanup() {
  echo "Beende Gazebo-Server ..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "${HEADLESS:-0}" == "1" ]]; then
  echo "Headless-Modus. Mit Ctrl+C beenden."
  wait "$SERVER_PID"
else
  # Dem Server kurz Zeit geben, die Welt zu laden.
  sleep 3
  echo "Starte Gazebo-GUI ..."
  gz sim -g -v 3
fi
