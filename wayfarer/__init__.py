"""Wayfarer: design a world's big shapes on purpose, then let the land form.

One world, in pipeline order (build.py runs it):
  rooms    the design: a graph of rooms (basins) and doors, as uplift, rock and water fields
  erode    land-forming: rivers cut, slopes fail, soil creeps
  style    regional character per room theme; rivers.py cuts the rivers
  place    settlements, landmarks and door roads by line of sight, then the measurements
  export   the game3d world directory: trees.py, biomes.py, paths.py and shots.py feed it
preview.py draws maps and software-rendered views for people.
"""
