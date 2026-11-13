#!/bin/sh
# Export one world, photograph its walls in game3d, and build the contact sheet.
#   ./shoot.sh <seed> <name> [key=value ...]      -> out/walls/<name>.jpg
set -e
cd "$(dirname "$0")"
seed=$1; name=$2; shift 2
pixi run python export_game3d.py "$seed" "$name" "$@" 2>&1 | grep -E "peak|wrote|Error|error" || true
( cd ../../game3d && WORLD_DIR=assets/terrain/$name TERRAIN_COVERAGE=4 WORLD_SHOTS=capture/$name \
    WORLD_SHOT_ONLY=${ONLY:-wall,overview} cargo run --release 2>&1 | grep -E "WORLD_SHOTS_OK|panicked|error" )
mkdir -p out/walls
pixi run python wsheet.py ../../game3d/capture/$name out/walls/$name.jpg
cp ../../game3d/assets/terrain/$name/map.png out/walls/${name}_map.png
