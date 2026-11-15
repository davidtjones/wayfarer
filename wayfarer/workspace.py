"""Where the sibling repos are.

Wayfarer writes worlds into the game3d app and reads its captures back. The app
is found once, here: `GAME3D_DIR` if set, else `../game3d` next to this repo.
A missing app is an error, not a silent write somewhere else.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this repo


def game3d_dir():
    """The game3d repo root, checked to hold a Cargo.toml."""
    path = os.path.abspath(os.environ.get("GAME3D_DIR") or os.path.join(ROOT, "..", "game3d"))
    if not os.path.isfile(os.path.join(path, "Cargo.toml")):
        raise FileNotFoundError(f"no game3d app at {path} (no Cargo.toml); set GAME3D_DIR to the game3d repo")
    return path


def world_dir(name):
    """Where an exported world named `name` lives inside game3d."""
    return os.path.join(game3d_dir(), "assets", "terrain", name)


def capture_dir(name):
    """Where game3d writes the captures of world `name` (`WORLD_SHOTS=capture/<name>`)."""
    return os.path.join(game3d_dir(), "capture", name)
