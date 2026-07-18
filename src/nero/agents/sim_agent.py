"""Simulation Agent: Test the ORB-SLAM navigation in simulation.

This agent runs the same navigation policy as orb_slam_agent but uses
a simulated robot and camera instead of physical hardware.

Agent/Policy Loop:
1. Show external camera stream of a space (both cameras)
2. Await a ``go to <object>`` direction and acknowledge it via speaker
3. Detect and track the requested object
4. Follow a dynamic world-frame goal pose derived from that object

Usage:
    uv run nero-sim
    uv run nero-sim --demo  # Run with pre-configured demo scene
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
from nero.interaction import TerminalCommandSource, request_navigation_target

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


class SimSpeaker:
    """Mock speaker for simulation."""

    def speak(self, text: str) -> None:
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
    commands = TerminalCommandSource()

    # Signal handler
    shutdown_event = False

    def signal_handler(sig, frame):
        nonlocal shutdown_event
        shutdown_event = True
        logger.info("Shutdown signal received")

    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    logger.info("Starting simulation loop (press Ctrl+C to stop)")
    loop_rate = 30
    loop_interval = 1.0 / loop_rate
    viz = Visualization()

    # State tracking
    target_object = None
    announced_arrival = False

    try:
        while not shutdown_event:
            loop_start = time.time()

            if target_object is None:
                try:
                    target_object = request_navigation_target(
                        speaker, commands, cancelled=lambda: shutdown_event
                    )
                except (EOFError, KeyboardInterrupt, InterruptedError):
                    shutdown_event = True
                    break
                policy.set_target(target_object)
                announced_arrival = False
                logger.info("Accepted navigation target: %s", target_object)

            # Get simulated camera frames (both RGB + Depth)
            frame = sim.get_frame()
            depth_frame = sim.get_depth_frame()
            if frame is None or depth_frame is None:
                time.sleep(0.01)
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

            # Display only detections relevant to the requested target.
            detections = [
                detection
                for detection in status.detections
                if target_object in detection.label.lower()
            ]
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
                    target_object = None
                    announced_arrival = False
                    logger.info("Reset - ready for new target")
                elif key == ord("d"):
                    sim.setup_demo_scene()
                    logger.info("Demo scene reloaded")

            # Check for completion
            if status.state == PolicyState.ARRIVED and not announced_arrival:
                logger.info(f"Arrived at {target_object}")
                speaker.speak(f"Arrived at {target_object}.")
                announced_arrival = True
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
        commands.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
