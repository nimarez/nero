"""Message-level readiness gate for the K1 Geek RGB-D camera."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


EXPECTED_WIDTH = 544
EXPECTED_HEIGHT = 448


def _stamp_seconds(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass
class CameraReadiness:
    """Track real messages rather than trusting DDS publisher discovery."""

    tolerance_seconds: float = 0.02
    minimum_pairs: int = 3
    minimum_fps: float = 10.0
    rgb_stamps: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    depth_stamps: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    camera_info_count: int = 0
    raw_rgb_count: int = 0
    incompatible: list[str] = field(default_factory=list)

    def observe(self, stream: str, message: Any) -> None:
        if stream in {"rgb", "depth", "camera_info"}:
            size = (int(message.width), int(message.height))
            if size != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
                issue = f"{stream} is {size[0]}x{size[1]}, expected 544x448"
                if issue not in self.incompatible:
                    self.incompatible.append(issue)
        if stream == "rgb":
            self.rgb_stamps.append(_stamp_seconds(message))
        elif stream == "depth":
            self.depth_stamps.append(_stamp_seconds(message))
        elif stream == "camera_info":
            self.camera_info_count += 1
        elif stream == "raw_rgb":
            self.raw_rgb_count += 1

    @property
    def synchronized_pairs(self) -> list[tuple[float, float]]:
        remaining_depth = list(self.depth_stamps)
        pairs = []
        for rgb in self.rgb_stamps:
            if not remaining_depth:
                break
            index = min(
                range(len(remaining_depth)),
                key=lambda candidate: abs(rgb - remaining_depth[candidate]),
            )
            depth = remaining_depth[index]
            if abs(rgb - depth) <= self.tolerance_seconds:
                pairs.append((rgb, depth))
                remaining_depth.pop(index)
        return pairs

    @staticmethod
    def _rate(stamps: deque[float]) -> float | None:
        if len(stamps) < 2:
            return None
        duration = max(stamps) - min(stamps)
        return None if duration <= 0 else (len(stamps) - 1) / duration

    @property
    def ready(self) -> bool:
        return (
            not self.incompatible
            and self.camera_info_count > 0
            and len(self.synchronized_pairs) >= self.minimum_pairs
            and (self._rate(self.rgb_stamps) or 0.0) >= self.minimum_fps
            and (self._rate(self.depth_stamps) or 0.0) >= self.minimum_fps
        )

    def summary(self, publisher_counts: dict[str, int] | None = None) -> str:
        publishers = publisher_counts or {}

        def state(name: str, count: int) -> str:
            if count:
                return f"{count} message(s)"
            discovered = publishers.get(name, 0)
            return f"no messages ({discovered} publisher(s) discovered)"

        details = [
            f"aligned RGB: {state('rgb', len(self.rgb_stamps))}",
            f"depth: {state('depth', len(self.depth_stamps))}",
            f"camera info: {state('camera_info', self.camera_info_count)}",
            f"raw RGB: {state('raw_rgb', self.raw_rgb_count)}",
        ]
        for name, stamps in (("RGB", self.rgb_stamps), ("depth", self.depth_stamps)):
            rate = self._rate(stamps)
            if rate is not None:
                details.append(f"{name} rate: {rate:.1f} FPS")
        if self.rgb_stamps and self.depth_stamps and not self.synchronized_pairs:
            closest = min(
                abs(rgb - depth) for rgb in self.rgb_stamps for depth in self.depth_stamps
            )
            details.append(f"closest RGB-D offset: {closest * 1000.0:.1f}ms")
        details.extend(self.incompatible)
        return "; ".join(details)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for live K1 RGB-D messages")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--report-interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.timeout <= 0 or args.sync_tolerance_ms < 0 or args.report_interval <= 0:
        parser.error("timeouts must be positive and synchronization tolerance non-negative")
    return args


def main() -> None:
    args = parse_args()
    try:
        import rclpy
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise SystemExit("K1 camera preflight requires the robot ROS 2 environment") from exc

    from nero.robot import K1Topics

    if not rclpy.ok():
        rclpy.init(args=None)
    node = rclpy.create_node("nero_k1_camera_preflight")
    topics = K1Topics()
    readiness = CameraReadiness(args.sync_tolerance_ms / 1000.0)
    qos = rclpy.qos.qos_profile_sensor_data
    _subscriptions = [
        node.create_subscription(Image, topics.rgb, lambda msg: readiness.observe("rgb", msg), qos),
        node.create_subscription(
            Image, topics.depth, lambda msg: readiness.observe("depth", msg), qos
        ),
        node.create_subscription(
            CameraInfo,
            topics.camera_info,
            lambda msg: readiness.observe("camera_info", msg),
            qos,
        ),
        node.create_subscription(
            Image, topics.raw_rgb, lambda msg: readiness.observe("raw_rgb", msg), qos
        ),
    ]
    deadline = time.monotonic() + args.timeout
    next_report = time.monotonic() + args.report_interval
    topic_names = {
        "rgb": topics.rgb,
        "depth": topics.depth,
        "camera_info": topics.camera_info,
        "raw_rgb": topics.raw_rgb,
    }
    try:
        while not readiness.ready and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() >= next_report:
                counts = {name: node.count_publishers(topic) for name, topic in topic_names.items()}
                print("Waiting for K1 RGB-D: " + readiness.summary(counts), flush=True)
                next_report = time.monotonic() + args.report_interval
        counts = {name: node.count_publishers(topic) for name, topic in topic_names.items()}
        if not readiness.ready:
            raise SystemExit(
                "K1 RGB-D preflight timed out: "
                + readiness.summary(counts)
                + ". Run `booster-cli launch -c restart -m perception`, wait for the robot "
                "to reconnect, then retry."
            )
        rgb_stamp, depth_stamp = readiness.synchronized_pairs[-1]
        print(
            "K1 RGB-D ready: 544x448 messages, "
            f"RGB-D offset={abs(rgb_stamp - depth_stamp) * 1000.0:.1f}ms",
            flush=True,
        )
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
