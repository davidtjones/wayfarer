#!/bin/sh
# Export one world, photograph its walls in game3d, and build the contact sheet.
#   pixi run shoot <seed> <name> [key=value ...]      -> out/walls/<name>.jpg
# game3d is $GAME3D_DIR, else ../game3d next to this repo (as in wayfarer/workspace.py).
set -e
cd "$(dirname "$0")/.."
seed=$1; name=$2; shift 2
g3d=${GAME3D_DIR:-../game3d}
[ -f "$g3d/Cargo.toml" ] || { echo "no game3d app at $g3d (no Cargo.toml); set GAME3D_DIR" >&2; exit 1; }
pixi run python -m wayfarer.export "$seed" "$name" "$@" 2>&1 | grep -E "peak|wrote|Error|error" || true
( cd "$g3d" && WORLD_DIR=assets/terrain/$name WORLD_SHOTS=capture/$name \
    WORLD_SHOT_ONLY=${ONLY:-wall,overview} cargo run --release 2>&1 | grep -E "WORLD_SHOTS_OK|panicked|error" )
mkdir -p out/walls
pixi run python -m tools.wall_sheet "$g3d/capture/$name" out/walls/$name.jpg
cp "$g3d/assets/terrain/$name/map.png" out/walls/${name}_map.png
