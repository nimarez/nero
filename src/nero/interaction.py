"""Command-driven human interaction shared by real and simulated agents."""

from __future__ import annotations

import logging
import os
import queue
import re
import socket
import threading
from pathlib import Path
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
    """Read an object name or full direction from a terminal."""

    accepts_bare_object_names = True

    def start_listening(self) -> None:
        return None

    def read_command(self, prompt: str) -> str:
        return input(prompt)

    def stop_listening(self) -> None:
        return None

    def close(self) -> None:
        return None


class UnixSocketCommandSource:
    """Receive deliberate object commands over a robot-local Unix socket."""

    accepts_bare_object_names = True

    def __init__(self, path: str | Path = "/tmp/nero-navigation.sock") -> None:
        self.path = Path(path)
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._closed = False

    def _open_server(self) -> None:
        if self._server is not None:
            return
        if self.path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.path))
            except OSError:
                self.path.unlink()
            else:
                raise RuntimeError(f"command socket is already in use: {self.path}")
            finally:
                probe.close()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.path))
            os.chmod(self.path, 0o600)
            server.listen(4)
            server.settimeout(0.25)
        except Exception:
            server.close()
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            raise
        self._server = server
        logger.info("Navigation command socket ready at %s", self.path)

    def start_listening(self) -> None:
        if self._closed:
            raise RuntimeError("navigation command socket is closed")
        self._open_server()

    def read_command(self, prompt: str) -> str:
        if self._server is None:
            raise RuntimeError("navigation command socket is not listening")
        logger.debug(prompt.rstrip())
        if self._client is None:
            try:
                self._client, _ = self._server.accept()
                self._client.settimeout(1.0)
            except socket.timeout:
                return ""
        try:
            payload = self._client.recv(4096)
        except socket.timeout:
            return ""
        if not payload:
            self._close_client()
            return ""
        return payload.decode("utf-8", errors="replace").splitlines()[0].strip()

    def acknowledge(self, response: str) -> None:
        """Reply only after the policy has semantically admitted a command."""
        if self._client is not None:
            try:
                self._client.sendall(f"{response}\n".encode())
            except OSError:
                pass
        self._close_client()

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def stop_listening(self) -> None:
        """Close admission completely so commands cannot queue while moving."""
        self._close_client()
        if self._server is not None:
            self._server.close()
            self._server = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_listening()


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


def _parse_bare_object_name(command: str) -> str | None:
    """Normalize a deliberately entered terminal target such as ``chair``."""
    object_name = " ".join(command.lower().split()).strip(" ,.!?")
    if not object_name or re.fullmatch(r"(?:please\s+)?go(?:\s+to)?", object_name):
        return None
    # Be forgiving in the deliberate terminal/socket UI: ``go the chair`` is a
    # common typo for ``go to the chair`` and must not become the literal prompt.
    object_name = re.sub(r"^(?:please\s+)?go(?:\s+to)?\s+", "", object_name)
    object_name = re.sub(r"^(?:the|a|an)\s+", "", object_name)
    return object_name or None


def request_navigation_target(
    speaker: Speaker,
    command_source: CommandSource,
    cancelled: Callable[[], bool] | None = None,
    target_validator: Callable[[str], bool] | None = None,
) -> str:
    """Wait for a valid direction, acknowledge it aloud, and return its target."""
    transcript = ""
    command_source.start_listening()
    try:
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("navigation command wait cancelled")
            accepts_bare_names = bool(
                getattr(command_source, "accepts_bare_object_names", False)
            )
            prompt = (
                "Object to follow (for example, 'chair'): "
                if accepts_bare_names
                else "Direction (for example, 'go to the chair'): "
            )
            command = command_source.read_command(prompt).strip()
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
            if object_name is None and accepts_bare_names:
                object_name = _parse_bare_object_name(command)
            if object_name is None:
                logger.info("Ignoring non-navigation command: %s", command)
                acknowledge = getattr(command_source, "acknowledge", None)
                if callable(acknowledge):
                    acknowledge("rejected")
                continue
            if target_validator is not None and not target_validator(object_name):
                logger.info("Rejecting unsupported object class: %s", object_name)
                acknowledge = getattr(command_source, "acknowledge", None)
                if callable(acknowledge):
                    acknowledge("unsupported")
                transcript = ""
                continue

            # Stop ASR before TTS so the K1 never transcribes its own
            # acknowledgement as the next human command.
            acknowledge = getattr(command_source, "acknowledge", None)
            if callable(acknowledge):
                acknowledge("accepted")
            command_source.stop_listening()
            try:
                speaker.speak(f"Going to the {object_name}.")
            except Exception:
                # Some K1 firmware exposes locomotion and sensors but not LUI TTS.
                # A missing announcement must not discard a deliberate command.
                logger.warning(
                    "Could not play navigation acknowledgement on the speaker",
                    exc_info=True,
                )
            return object_name
    finally:
        command_source.stop_listening()


class NavigationTargetListener:
    """Acquire a human direction without blocking sensing or visualization."""

    def __init__(
        self,
        speaker: Speaker,
        command_source: CommandSource,
        *,
        cancelled: Callable[[], bool] | None = None,
        target_validator: Callable[[str], bool] | None = None,
    ) -> None:
        self._speaker = speaker
        self._command_source = command_source
        self._cancelled = cancelled
        self._target_validator = target_validator
        self._results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._listen,
            name="nero-navigation-command",
            daemon=True,
        )
        self._thread.start()

    def _listen(self) -> None:
        try:
            target = request_navigation_target(
                self._speaker,
                self._command_source,
                cancelled=self._cancelled,
                target_validator=self._target_validator,
            )
            self._results.put(("target", target))
        except BaseException as exc:
            self._results.put(("error", exc))

    def poll(self) -> str | None:
        """Return a completed target, or ``None`` while input is pending."""
        try:
            kind, value = self._results.get_nowait()
        except queue.Empty:
            return None
        if kind == "error":
            raise value
        return str(value)

    def close(self) -> None:
        self._command_source.close()


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
