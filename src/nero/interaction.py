"""Human interaction helpers shared by real and simulated agents."""

from __future__ import annotations

from typing import Callable, Protocol


class Speaker(Protocol):
    """Minimal speech interface used by object-following agents."""

    def speak(self, text: str) -> None: ...


def announce_and_confirm(
    speaker: Speaker,
    object_name: str,
    confirm: Callable[[str], str] | None = None,
) -> bool:
    """Announce a detected object and wait for explicit human consent."""
    message = f"{object_name} detected. Should I follow it?"
    speaker.speak(message)
    ask = confirm or input
    response = ask(f"{message} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def safe_stand_off_distance(object_name: str) -> float:
    """Choose an internal safety radius from the detected object class."""
    furniture = {"table", "desk", "couch", "sofa", "bed", "chair", "cabinet", "shelf"}
    small_objects = {"bottle", "cup", "phone", "keys", "book", "lamp", "plant"}
    normalized_name = object_name.lower()

    if normalized_name in furniture:
        return 1.0
    if normalized_name in small_objects:
        return 0.7
    return 0.8
