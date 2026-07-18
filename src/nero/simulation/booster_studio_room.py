"""Install Nero's furnished-room scene into a Booster Studio simulator image."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

SCENE_DIR = Path(__file__).with_name("scenes") / "booster_studio_living_room"
SCENE_NAME = "nero_living_room_K1"
DEFAULT_SCENE_NAME = "default_pitch_K1"
BACKUP_SUFFIX = ".nero-original"


def find_sim_root(explicit: str | Path | None = None) -> Path:
    """Find a Booster Studio ``robocup_sim_src`` directory."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if env_root := os.environ.get("BOOSTER_STUDIO_SIM_ROOT"):
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path.cwd(),
            Path.home() / "robocup_sim_src",
            Path("/opt/booster/robocup_sim_src"),
            Path(
                "/Applications/Booster Studio.app/Contents/Resources/app/booster-native/"
                "statics/virtual-robot/robocup_sim_src"
            ),
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "mjcf" / "K1_22dof.xml").is_file():
            return candidate
        if candidate.name == "mjcf" and (candidate / "K1_22dof.xml").is_file():
            return candidate.parent
    raise FileNotFoundError(
        "Could not find Booster Studio's robocup_sim_src. Pass --sim-root or set "
        "BOOSTER_STUDIO_SIM_ROOT."
    )


def _copy_scene(sim_root: Path) -> tuple[Path, Path]:
    mjcf_dir = sim_root / "mjcf"
    scene_target = mjcf_dir / f"{SCENE_NAME}.xml"
    extensions_target = mjcf_dir / f"{SCENE_NAME}.extensions.xml"
    shutil.copy2(SCENE_DIR / "living_room_K1.xml", scene_target)
    shutil.copy2(SCENE_DIR / "living_room_K1.extensions.xml", extensions_target)
    shutil.copytree(
        SCENE_DIR / "assets" / "nero_room",
        mjcf_dir / "assets" / "nero_room",
        dirs_exist_ok=True,
    )
    return scene_target, extensions_target


def install_room(sim_root: Path, *, activate: bool = False) -> tuple[Path, Path]:
    """Copy the room into a simulator tree and optionally replace its empty scene."""
    scene_target, extensions_target = _copy_scene(sim_root)
    if not activate:
        return scene_target, extensions_target

    if ".app/Contents/" in str(sim_root):
        raise PermissionError(
            "Refusing to modify the signed macOS application bundle. Run this command "
            "inside the disposable Booster Studio virtual-robot terminal instead."
        )

    mjcf_dir = sim_root / "mjcf"
    pairs = [
        (scene_target, mjcf_dir / f"{DEFAULT_SCENE_NAME}.xml"),
        (extensions_target, mjcf_dir / f"{DEFAULT_SCENE_NAME}.extensions.xml"),
    ]
    for source, target in pairs:
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
    return scene_target, extensions_target


def restore_default_scene(sim_root: Path) -> None:
    """Restore the original empty K1 scene after an activated room test."""
    mjcf_dir = sim_root / "mjcf"
    restored = 0
    for filename in (
        f"{DEFAULT_SCENE_NAME}.xml",
        f"{DEFAULT_SCENE_NAME}.extensions.xml",
    ):
        target = mjcf_dir / filename
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        if backup.is_file():
            shutil.copy2(backup, target)
            restored += 1
    if restored != 2:
        raise FileNotFoundError(
            "A complete Nero scene backup was not found; nothing restored"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Nero's CC0 living-room scene into Booster Studio"
    )
    parser.add_argument("--sim-root", help="Path to Booster Studio's robocup_sim_src")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--activate",
        action="store_true",
        help="Back up and replace the simulator's empty K1 scene inside its container",
    )
    action.add_argument(
        "--restore",
        action="store_true",
        help="Restore the empty K1 scene backed up by --activate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_root = find_sim_root(args.sim_root)
    if args.restore:
        restore_default_scene(sim_root)
        print(f"Restored Booster Studio's empty K1 scene in {sim_root}")
        return
    scene, _ = install_room(sim_root, activate=args.activate)
    status = "installed and activated" if args.activate else "installed"
    print(f"Nero living room {status}: {scene}")
    if args.activate:
        print(
            "Restart the virtual robot or switch away from and back to the empty K1 scene."
        )


if __name__ == "__main__":
    main()
