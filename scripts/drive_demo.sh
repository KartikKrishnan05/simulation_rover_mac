#!/usr/bin/env bash
# Kleine Demo-Fahrt ohne Nav2: geradeaus nach Westen, Wende in einem Zug
# (Ackermann – kein Drehen auf der Stelle) und zurück.
# Simulation muss laufen (./run.sh).
set -euo pipefail

export GZ_IP="${GZ_IP:-127.0.0.1}"
TOPIC="${TOPIC:-/cmd_vel}"

drive() { # linear angular sekunden
  echo "linear=$1 angular=$2 für $3 s"
  gz topic -t "$TOPIC" -m gz.msgs.Twist -p "linear: {x: $1}, angular: {z: $2}"
  sleep "$3"
}

trap 'drive 0 0 0' EXIT INT TERM

drive 0.6 0.0 5      # nach Westen
drive 0.4 -0.46 6.8  # Rechtskurve mit minimalem Radius (≈ 0,87 m) → 180°
drive 0.6 0.0 5      # zurück nach Osten
