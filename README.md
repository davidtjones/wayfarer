# Wayfarer

Wayfarer designs the big shapes of a world on purpose and lets simulated
land-forming make them read natural. A world is a graph of rooms and doors:
basins with a floor and a theme, separated by walls (mountain country, scarps,
hills, water), joined by doors where a road crosses a wall. Settlements,
landmarks and roads are then placed by what the player can see, the result is
measured, and the world is exported as a directory that the `game3d` app loads.

Python 3.12 via pixi (numpy, scipy, numba, pillow).

## Running it

Run from this repo's root. Arguments after a task name are passed through.

```sh
pixi install

pixi run build 3 rooms=9 --map=out/map.png   # build one world, print its measurements
pixi run export 3 rooms3 rooms=9             # write it to <game3d>/assets/terrain/rooms3/
pixi run previews 3 out/prev3 rooms=9        # software-rendered views, one walk per door
pixi run sweep out/rooms/sweep.json          # vary one design knob at a time, three seeds each
pixi run sweep-table out/rooms/sweep.json    # one row per arm of a sweep
pixi run shoot 3 rooms3 rooms=9              # export, photograph the walls in game3d, contact sheet
pixi run gallery rooms3 [more names]         # page of game3d captures -> out/rooms/index.html
```

`key=value` arguments set fields of `Cfg` in `wayfarer/rooms.py` (`rooms`,
`wall`, `loops`, `hub`, `belt`, ...). See a world in the game with
`WORLD_DIR=assets/terrain/<name> cargo run --release` in `game3d`.

**Where game3d is.** `export`, `shoot` and `gallery` find the app through
`wayfarer/workspace.py`: `GAME3D_DIR` if set, else `../game3d` next to this
repo. They stop with an error if there is no `Cargo.toml` there.

## Layout

One flat package. The files are listed in pipeline order.

```
wayfarer/
  build.py       one world end to end: rooms -> erode -> style -> place -> measure
  rooms.py       the design: a graph of rooms and doors, as uplift, rock and water fields
  erode.py       land-forming: rivers cut, slopes fail, soil creeps; flow routing
  style.py       regional character per room theme, lakes settled, grid refined
  rivers.py      river beds cut where the land already sends water, and their current
  place.py       line of sight and routing; settlements, landmarks, door roads; measurements
  paths.py       roads, footpaths and trails as a network and a distance map
  biomes.py      what lives where, from biomes.toml
  biomes.toml    the biome table, per room theme
  trees.py       where trees grow and which species
  export.py      write a world directory for game3d
  shots.py       the shot list game3d photographs (walls, lakes, rivers)
  preview.py     the diagnostic map, and software-rendered views of a dressed world
  noise.py       seeded value noise, smoothstep
  workspace.py   where game3d is
tools/           scripts that drive the package: previews, sweep, sweep_table,
                 gallery, door_sheet, wall_sheet, shoot.sh
out/             generated output (gitignored)
```

Each module's docstring is its reference.

## Limits

The preview renderer is for judging shapes quickly. It is not how the game draws
the world.
