#!/usr/bin/env python3
"""Simulation Agent: Test the ORB-SLAM navigation in simulation.

This agent runs the same navigation policy as orb_slam_agent but uses
a simulated robot and camera instead of physical hardware.

Agent/Policy Loop:
1. Show external camera stream of a space (both cameras)
2. Detect objects in real-time and announce them via speaker
3. Await user confirmation to follow a detected object
4. Navigate to the confirmed object (target distance deduced by robot)

Usage:
    python -m nero.agents.sim_agent
    python -m nero.agents.sim_agent --demo  # Run with pre-configured demo scene
"""

from __future__ import annotations

import argparse
import logging
import signal
import time

import cv2

from nero.simulation.environment import SimEnvironment
from nero.simulation.sim_camera import CameraMode
from nero.utils.visualization import Visualization
from nero.navigation.policy import NavigationPolicy, PolicyState

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nero Simulation Agent - Test navigation in sim"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with pre-configured demo scene",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable visual display",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def announce_object(speaker, object_name: str, distance: float) -> bool:
    """Announce detected object via speaker and await user confirmation.

    Args:
        speaker: Speech synthesis interface (or mock for sim)
        object_name: Name of detected object
        distance: Distance to object in meters

    Returns:
        True if user confirms, False otherwise
    """
    message = f"I see a {object_name} at {distance:.1f} meters. Should I go to it?"
    logger.info(f"Announcing: {message}")

    # In sim mode, just log and prompt
    if speaker is None:
        print(f"\n>>> {message}")
        response = input("Confirm (y/n): ").strip().lower()
        return response in ("y", "yes")

    # Real robot: use speaker
    try:
        speaker.synthesize(message)
        # Wait for user response via voice or button
        return True  # Placeholder - would need voice input or button press
    except Exception as e:
        logger.error(f"Speaker announcement failed: {e}")
        return False


def deduce_target_distance(object_name: str, distance: float) -> float:
    """Deduce appropriate target distance based on object type and current distance.

    The robot determines how close to get based on:
    - Object type (furniture vs small objects)
    - Current distance to object
    - Safety margins

    Args:
        object_name: Name of the object
        distance: Current distance to object

    Returns:
        Target distance in meters
    """
    # Furniture - stop further away
    furniture = {"table", "desk", "couch", "sofa", "bed", "chair", "cabinet", "shelf"}
    # Small objects - get closer
    small_objects = {"bottle", "cup", "phone", "keys", "book", "lamp", "plant"}

    obj_lower = object_name.lower()

    if obj_lower in furniture:
        return max(1.0, distance * 0.5)  # Stop at 1m or half current distance
    elif obj_lower in small_objects:
        return max(0.5, distance * 0.3)  # Get closer for small objects
    else:
        return max(0.8, distance * 0.4)  # Default


class SimSpeaker:
    """Mock speaker for simulation."""

    def synthesize(self, text: str) -> None:
        """Mock speech synthesis - just print."""
        print(f"\n[SPEAKER] {text}")


def main():
    """Main entry point for simulation agent."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting Nero Simulation Agent")
    logger.info("Using both cameras (RGB + Depth)")

    # Initialize simulation environment
    sim = SimEnvironment(
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        camera_mode=CameraMode.TOP_DOWN,
    )
    # Set up demo scene if requested
    if args.demo:
        sim.setup_demo_scene()
        logger.info("Demo scene loaded")

    # Initialize navigation policy (same policy, just sim environment)
    policy = NavigationPolicy(
        sim_env=sim,
    )
    policy.start()

    # Mock speaker for sim
    speaker = SimSpeaker()

    # Signal handler
    shutdown_event = False

    def signal_handler(sig, frame):
        nonlocal shutdown_event
        shutdown_event = True
        logger.info("Shutdown signal received")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    logger.info("Starting simulation loop (press Ctrl+C to stop)")
    loop_rate = 30
    loop_interval = 1.0 / loop_rate
    viz = Visualization()

    # State tracking
    confirmed_object = None
    target_distance = None
    announced_objects = set()  # Track what we've already announced
    announce_cooldown = 5.0  # seconds between re-announcing same object
    last_announce_time = {}

    try:
        while not shutdown_event:
            loop_start = time.time()

            # Get simulated camera frames (both RGB + Depth)
            frame = sim.get_frame()
            depth_frame = sim.get_depth_frame()
            if frame is None or depth_frame is None:
                time.sleep(0.01)
                continue

            # Detect objects in real-time
            detections = sim.get_detections()

            # Check for new objects to announce
            current_time = time.time()
            for detection in detections:
                obj_name = detection.label.lower()
                if obj_name in announced_objects:
                    # Check cooldown
                    if (
                        current_time - last_announce_time.get(obj_name, 0)
                        < announce_cooldown
                    ):
                        continue

                # Announce and await confirmation
                should_follow = announce_object(speaker, obj_name, detection.distance)

                if should_follow:
                    confirmed_object = obj_name
                    target_distance = deduce_target_distance(
                        obj_name, detection.distance
                    )
                    logger.info(
                        f"Confirmed target: {confirmed_object} at {target_distance:.2f}m"
                    )
                    policy.set_target(confirmed_object)
                    policy._goal.target_distance = target_distance
                    announced_objects.add(obj_name)
                    last_announce_time[obj_name] = current_time
                    break  # Only handle one object at a time

            # Get current policy state
            status = policy.status

            # If no target confirmed yet, show idle state
            if confirmed_object is None and status.state == PolicyState.IDLE:
                frame_with_text = viz.draw_navigation_info(
                    frame,
                    state="idle",
                    message="Scanning for objects... (press 'd' for demo scene)",
                    fps=sim.camera.get_fps(),
                )
                if not args.no_display:
                    key = viz.show_stream(
                        frame_with_text, "Nero Simulation", sim.camera.get_fps()
                    )
                    if key == ord("q"):
                        shutdown_event = True
                    elif key == ord("d"):
                        sim.setup_demo_scene()
                        announced_objects.clear()
                        logger.info("Demo scene reloaded")
                continue

            # Step policy
            status = policy.step()

            # Draw overlay with both camera info
            frame = viz.draw_navigation_info(
                frame,
                state=status.state.value,
                message=status.message,
                fps=sim.camera.get_fps(),
                velocity=(
                    (
                        status.velocity_command.linear_x,
                        status.velocity_command.angular_z,
                    )
                    if status.velocity_command
                    else None
                ),
            )

            # Display detected objects
            if detections:
                obj_text = ", ".join(
                    [f"{d.label}({d.distance:.1f}m)" for d in detections]
                )
                cv2.putText(
                    frame,
                    f"Detected: {obj_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            # Display
            if not args.no_display:
                key = viz.show_stream(frame, "Nero Simulation", sim.camera.get_fps())
                if key == ord("q"):
                    shutdown_event = True
                elif key == ord("r") and status.state == PolicyState.ARRIVED:
                    # Reset and look for new object
                    policy.reset()
                    confirmed_object = None
                    target_distance = None
                    announced_objects.clear()
                    logger.info("Reset - ready for new target")
                elif key == ord("d"):
                    sim.setup_demo_scene()
                    announced_objects.clear()
                    logger.info("Demo scene reloaded")

            # Check for completion
            if status.state == PolicyState.ARRIVED:
                logger.info(f"Arrived at {confirmed_object}")
                if not args.no_display:
                    logger.info(
                        "Press 'r' to reset and find another object, 'q' to quit"
                    )

            # Maintain loop rate
            elapsed = time.time() - loop_start
            if elapsed < loop_interval:
                time.sleep(loop_interval - elapsed)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    finally:
        # Cleanup
        logger.info("Shutting down...")
        policy.stop()
        sim.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
