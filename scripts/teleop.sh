#!/usr/bin/env bash
# Terminal-Fernsteuerung für den Rover (Alternative zur GUI).
# Sendet gz.msgs.Twist auf /cmd_vel direkt an Gazebo (ohne ROS).
#
# Ackermann-Lenkung: angular.z ist die gewünschte Gierrate. Der Rover lenkt
# entsprechend ein, dreht sich aber nur, wenn er fährt (Wendekreis ≈ 0,87 m).
#
#   w / s   schneller vorwärts / rückwärts
#   a / d   stärker nach links / rechts lenken
#   x       Lenkung gerade
#   Leertaste  Stopp
#   q       Beenden
set -euo pipefail

export GZ_IP="${GZ_IP:-127.0.0.1}"
TOPIC="${TOPIC:-/cmd_vel}"
LIN_STEP=0.2
ANG_STEP=0.2
LIN_MAX=1.0
ANG_MAX=1.2

lin=0.0
ang=0.0

clamp() { # wert max
  awk -v v="$1" -v m="$2" 'BEGIN { if (v > m) v = m; if (v < -m) v = -m; printf "%.2f", v }'
}

send() {
  gz topic -t "$TOPIC" -m gz.msgs.Twist -p "linear: {x: $lin}, angular: {z: $ang}" >/dev/null 2>&1 &
}

stty_saved="$(stty -g)"
restore() {
  stty "$stty_saved"
  lin=0.0; ang=0.0; send; wait
  echo
}
trap restore EXIT INT TERM
stty -echo -icanon min 1 time 0

echo "Rover-Teleop auf $TOPIC  (w/s/a/d, x, Leertaste = Stopp, q = Ende)"
while true; do
  key="$(dd bs=1 count=1 2>/dev/null)"
  case "$key" in
    w) lin="$(clamp "$(awk -v a="$lin" -v b=$LIN_STEP 'BEGIN{print a+b}')" $LIN_MAX)" ;;
    s) lin="$(clamp "$(awk -v a="$lin" -v b=$LIN_STEP 'BEGIN{print a-b}')" $LIN_MAX)" ;;
    a) ang="$(clamp "$(awk -v a="$ang" -v b=$ANG_STEP 'BEGIN{print a+b}')" $ANG_MAX)" ;;
    d) ang="$(clamp "$(awk -v a="$ang" -v b=$ANG_STEP 'BEGIN{print a-b}')" $ANG_MAX)" ;;
    x) ang=0.0 ;;
    " ") lin=0.0; ang=0.0 ;;
    q) break ;;
    *) continue ;;
  esac
  printf "\rlinear.x = %5s m/s   angular.z = %5s rad/s   " "$lin" "$ang"
  send
done
