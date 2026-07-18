"""Command-driven human interaction shared by real and simulated agents."""

from __future__ import annotations

import logging
import os
import queue
import re
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class Speaker(Protocol):
    """Minimal speech output used by object-navigation agents."""

    def speak(self, text: str) -> None: ...


class CommandSource(Protocol):
    """Source of spoken or typed natural-language directions."""

    def start_listening(self) -> None: ...

    def read_command(self, prompt: str) -> str: ...

    def stop_listening(self) -> None: ...

    def close(self) -> None: ...


class TerminalCommandSource:
    """Read directions from a terminal, primarily for simulation."""

    def start_listening(self) -> None:
        return None

    def read_command(self, prompt: str) -> str:
        return input(prompt)

    def stop_listening(self) -> None:
        return None

    def close(self) -> None:
        return None


class K1VoiceCommandSource:
    """Read transcribed speech from the K1's official LUI ASR service."""

    def __init__(self, robot_name: str = "", network_interface: str | None = None):
        try:
            import booster_robotics_sdk_python as booster
        except ImportError as exc:
            raise RuntimeError(
                "K1 voice commands require the Booster Robotics SDK"
            ) from exc

        self._commands: queue.Queue[str] = queue.Queue(maxsize=64)
        self._closed = False
        self._listening = False
        self._network_interface = network_interface or os.getenv("BOOSTER_NET_IF", "lo")
        booster.ChannelFactory.Instance().Init(0, self._network_interface)
        self._client = booster.LuiClient()
        self._subscriber = booster.LuiAsrChunkSubscriber(self._on_asr_chunk)

        try:
            if robot_name:
                self._client.InitWithName(robot_name)
                self._subscriber.InitChannelWithName(robot_name)
            else:
                self._client.Init()
                self._subscriber.InitChannel()
        except Exception:
            try:
                self._subscriber.CloseChannel()
            except Exception:
                logger.debug("Could not close partially initialized ASR channel")
            self._closed = True
            raise

    def start_listening(self) -> None:
        if self._closed:
            raise RuntimeError("K1 voice command source is closed")
        if self._listening:
            return
        while not self._commands.empty():
            try:
                self._commands.get_nowait()
            except queue.Empty:
                break
        self._client.StartAsr()
        self._listening = True
        logger.info(
            "Listening for K1 voice commands on interface %s",
            self._network_interface,
        )

    def _on_asr_chunk(self, chunk: Any) -> None:
        text = str(getattr(chunk, "text", "")).strip()
        if text:
            logger.info("K1 heard: %s", text)
            try:
                self._commands.put_nowait(text)
            except queue.Full:
                # Prefer fresh speech over stale partial transcripts.
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._commands.put_nowait(text)
                except queue.Full:
                    pass

    def read_command(self, prompt: str) -> str:
        if not self._listening:
            raise RuntimeError("K1 ASR is not listening")
        logger.info(prompt.rstrip())
        try:
            return self._commands.get(timeout=0.25)
        except queue.Empty:
            return ""

    def stop_listening(self) -> None:
        if not self._listening:
            return
        try:
            self._client.StopAsr()
        finally:
            self._listening = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop_listening()
        finally:
            self._subscriber.CloseChannel()


_GO_TO_PATTERN = re.compile(
    r"^\s*(?:please\s+)?go\s+to\s+(?:the\s+|a\s+|an\s+)?"
    r"(?P<object>.+?)(?:\s*,?\s*please)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def parse_go_to_command(command: str) -> str | None:
    """Extract the requested object from a ``go to <object>`` direction."""
    match = _GO_TO_PATTERN.match(command)
    if match is None:
        return None
    object_name = " ".join(match.group("object").lower().split()).strip(" ,.!?")
    return object_name or None


def request_navigation_target(
    speaker: Speaker,
    command_source: CommandSource,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    """Wait for a valid direction, acknowledge it aloud, and return its target."""
    transcript = ""
    command_source.start_listening()
    try:
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("navigation command wait cancelled")
            command = command_source.read_command(
                "Direction (for example, 'go to the chair'): "
            ).strip()
            if not command:
                continue

            # Some ASR runtimes emit an utterance in multiple chunks. Keep a short
            # rolling transcript so "go to" followed by "the chair" still parses.
            transcript = f"{transcript} {command}".strip()
            words = transcript.split()
            transcript = " ".join(words[-20:])
            object_name = parse_go_to_command(command) or parse_go_to_command(
                transcript
            )
            if object_name is None:
                logger.info("Ignoring non-navigation command: %s", command)
                continue

            # Stop ASR before TTS so the K1 never transcribes its own
            # acknowledgement as the next human command.
            command_source.stop_listening()
            speaker.speak(f"Going to the {object_name}.")
            return object_name
    finally:
        command_source.stop_listening()


def safe_stand_off_distance(object_name: str) -> float:
    """Choose an internal safety radius from the requested object class."""
    furniture = {"table", "desk", "couch", "sofa", "bed", "chair", "cabinet", "shelf"}
    small_objects = {"bottle", "cup", "phone", "keys", "book", "lamp", "plant"}
    normalized_name = object_name.lower()

    if normalized_name in furniture:
        return 1.0
    if normalized_name in small_objects:
        return 0.7
    return 0.8
