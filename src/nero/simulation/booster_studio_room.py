"""Install Nero's furnished-room scene into a Booster Studio simulator image."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from nero.slam.k1_calibration import K1Calibration

SCENE_DIR = Path(__file__).with_name("scenes") / "booster_studio_living_room"
SCENE_NAME = "nero_living_room_K1"
DEFAULT_SCENE_NAME = "default_pitch_K1"
BACKUP_SUFFIX = ".nero-original"
CALIBRATED_MODEL_NAME = "K1_22dof_nero.xml"
DEFAULT_SENSOR_CALIBRATION = Path("config/k1_geek_nominal_calibration.json")


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
            Path("/usr/local/booster_robot/booster_robocup_sim"),
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


def _write_calibrated_k1_model(sim_root: Path, calibration: K1Calibration) -> Path:
    """Copy the vendor K1 model and give its renderer the real camera profile."""
    source = sim_root / "mjcf" / "K1_22dof.xml"
    target = sim_root / "mjcf" / CALIBRATED_MODEL_NAME
    tree = ET.parse(source)
    camera = tree.getroot().find(".//camera[@name='rgbd_camera']")
    if camera is None:
        raise RuntimeError(f"K1 RGB-D camera was not found in {source}")
    k = calibration.camera_matrix
    camera.set("resolution", f"{calibration.width} {calibration.height}")
    # MuJoCo's K1 renderer accepts vertical FOV. Derive it from the robot's
    # calibrated fy instead of retaining Studio's unrelated 58-degree default.
    fovy = 2.0 * math.degrees(math.atan(float(calibration.height) / (2.0 * float(k[4]))))
    camera.set("fovy", f"{fovy:.12g}")
    tree.write(target, encoding="unicode")
    return target


def _copy_scene(sim_root: Path, calibration: K1Calibration) -> tuple[Path, Path]:
    mjcf_dir = sim_root / "mjcf"
    scene_target = mjcf_dir / f"{SCENE_NAME}.xml"
    extensions_target = mjcf_dir / f"{SCENE_NAME}.extensions.xml"
    _write_calibrated_k1_model(sim_root, calibration)
    scene_text = (
        (SCENE_DIR / "living_room_K1.xml")
        .read_text()
        .replace('file="K1_22dof.xml"', f'file="{CALIBRATED_MODEL_NAME}"')
    )
    scene_target.write_text(scene_text)
    shutil.copy2(SCENE_DIR / "living_room_K1.extensions.xml", extensions_target)
    shutil.copytree(
        SCENE_DIR / "assets" / "nero_room",
        mjcf_dir / "assets" / "nero_room",
        dirs_exist_ok=True,
    )
    return scene_target, extensions_target


def configure_ros_transport(container_env: Path) -> None:
    """Back up the container boot config and enable its ROS sensor transport."""
    backup = container_env.with_name(container_env.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(container_env, backup)
    lines = container_env.read_text().splitlines()
    updated = []
    found = False
    for line in lines:
        if line.startswith("SIM_TRANSPORT="):
            updated.append("SIM_TRANSPORT=ros")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append("SIM_TRANSPORT=ros")
    container_env.write_text("\n".join(updated) + "\n")


def install_room(
    sim_root: Path,
    *,
    activate: bool = False,
    container_env: Path | None = None,
    sensor_calibration: Path = DEFAULT_SENSOR_CALIBRATION,
) -> tuple[Path, Path]:
    """Copy the room into a simulator tree and optionally replace its empty scene."""
    if ".app/Contents/" in str(sim_root):
        raise PermissionError(
            "Refusing to modify the signed macOS application bundle. Run this command "
            "inside the disposable Booster Studio virtual-robot terminal instead."
        )
    calibration = K1Calibration.load(sensor_calibration)
    calibration.validate_geek_profile()
    scene_target, extensions_target = _copy_scene(sim_root, calibration)
    if not activate:
        return scene_target, extensions_target

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
    if container_env is not None:
        configure_ros_transport(container_env)
    return scene_target, extensions_target


def restore_default_scene(sim_root: Path, *, container_env: Path | None = None) -> None:
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
        raise FileNotFoundError("A complete Nero scene backup was not found; nothing restored")
    if container_env is not None:
        backup = container_env.with_name(container_env.name + BACKUP_SUFFIX)
        if not backup.is_file():
            raise FileNotFoundError("The original simulator transport config was not found")
        shutil.copy2(backup, container_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Nero's CC0 living-room scene into Booster Studio"
    )
    parser.add_argument("--sim-root", help="Path to Booster Studio's robocup_sim_src")
    parser.add_argument(
        "--sensor-calibration",
        type=Path,
        default=DEFAULT_SENSOR_CALIBRATION,
        help="Real K1 calibration; defaults to Nero's nominal K1 Geek profile",
    )
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
    container_env = Path.home() / ".env"
    if not container_env.is_file():
        container_env = None
    if args.restore:
        restore_default_scene(sim_root, container_env=container_env)
        print(f"Restored Booster Studio's empty K1 scene in {sim_root}")
        return
    scene, _ = install_room(
        sim_root,
        activate=args.activate,
        container_env=container_env if args.activate else None,
        sensor_calibration=args.sensor_calibration,
    )
    status = "installed and activated" if args.activate else "installed"
    print(f"Nero living room {status}: {scene}")
    if args.activate:
        print("Restart the virtual robot or switch away from and back to the empty K1 scene.")


if __name__ == "__main__":
    main()
