#!/usr/bin/env bash
# Installiert Gazebo Harmonic über Homebrew.
# Achtung: Die Pakete werden ggf. aus dem Quellcode gebaut – das kann
# je nach Mac 30–90 Minuten dauern.
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew fehlt. Installation siehe https://brew.sh" >&2
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools fehlen. Bitte 'xcode-select --install' ausführen." >&2
  exit 1
fi

GZ_RELEASE="${GZ_RELEASE:-gz-harmonic}"

brew tap osrf/simulation
# Homebrew ab Version 7 lädt Formeln aus Fremd-Taps nur nach Freigabe.
brew trust osrf/simulation 2>/dev/null || true
brew update
brew install "$GZ_RELEASE"

echo
echo "Fertig. Test: gz sim --versions"
gz sim --versions || true
