# Project notes for agents - `wayfarer`

**Read `../NOW.md` first** - David's current direction, which overrides this file.

Before working in this repo, read `../AGENTS.md` for shared workspace guidance
unless it is already in your context. Resolve that path relative to this repo's
root, even when working in a subdirectory.

`AGENTS.md` is canonical; `CLAUDE.md` is a relative symlink to it for Claude
compatibility. Edit `AGENTS.md` and keep the symlink intact.

This is one repo in the `game3d-ws` workspace. For the workspace layout, the
shared work journal, and the cross-repo documentation protocol, see the
workspace-level notes one directory up: `../AGENTS.md`. This file covers
Wayfarer specifically.

## What this is

**Wayfarer** is a world generator in Python 3.12 (via **pixi**). It designs a
world as rooms and doors, forms the land with erosion, places sites by line of
sight, measures the result, and **exports a world directory that the sibling
`game3d` app loads** (`game3d/src/world/mod.rs` reads it). `README.md` covers
what it does, how to run it, and the layout map; read it before changing
anything.

## Layout

- `wayfarer/` - the package, flat: one module per job, listed in pipeline order
  in `README.md`. `build.py` runs one world end to end; `export.py` writes it for
  game3d. Keep it flat and keep module names unique and descriptive.
- `wayfarer/workspace.py` - **the only place that knows where `game3d` is**
  (`GAME3D_DIR`, else `../game3d`). Anything that reads or writes inside the app
  goes through it; never write a relative `../game3d` path elsewhere.
  `tools/shoot.sh` repeats the same default in shell.
- `tools/` - scripts that drive the package (previews, sweeps, galleries,
  contact sheets, `shoot.sh`). Not imported by the package.
- `out/` - generated output. **gitignored.**
- `pixi.toml` - the env and the tasks. pixi stays (not uv): world generation may
  later need ML or data-science packages from conda-forge.

Each module's docstring is its reference. Inside the package, imports are
relative (`from .erode import form`); tools import `wayfarer.<module>`. Entry
points run as modules (`python -m wayfarer.export`, `python -m tools.sweep`)
from the repo root, which the pixi tasks do.

## Documentation

The shared journal (`../journal/`), the experiment registry (`../experiments/`),
and the cross-repo documentation protocol are workspace-level; see
`../AGENTS.md`. This repo has no status doc.

**When you change X, update Y** (the workspace-wide rules in `../AGENTS.md` also
apply):

- A module's behaviour -> its docstring.
- Layout, tasks, commands or dependencies -> `README.md` and this file.
- The world directory format (`export.py`) -> its docstring, and check
  `../game3d/src/world/mod.rs` still reads it.

## Common commands

Run from this repo root.

- `pixi run build <seed> [key=value ...] [--map=out.png]` - build one world and print its measurements.
- `pixi run export <seed> <name> [key=value ...]` - write a world to `<game3d>/assets/terrain/<name>/`.
- `pixi run previews <seed> <out_dir> [key=value ...]` - software-rendered views of a world.
- `pixi run sweep <out.json>`, `pixi run sweep-table <out.json>` - vary one design knob at a time, then tabulate.
- `pixi run shoot <seed> <name> [key=value ...]` - export, photograph the walls in game3d, build the contact sheet.
- `pixi run gallery <name> [...]` - collect game3d captures into `out/rooms/index.html`.

The art world `game3d` loads by default is `pixi run export 0 wayfarer-art rooms=9`
(no other options: `n=` changes the layout).
