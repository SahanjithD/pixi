from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Optional, Sequence

ActionHandler = Callable[..., None]


class ActionName(str, Enum):
    """All high-level actions Pixi can perform."""
    GREET_HAPPILY = "GREET_HAPPILY"
    FOLLOW_PERSON = "FOLLOW_PERSON"
    TILT_HEAD_CURIOUSLY = "TILT_HEAD_CURIOUSLY"
    DO_A_HAPPY_DANCE = "DO_A_HAPPY_DANCE"
    LOOK_AROUND = "LOOK_AROUND"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    GO_TO_SLEEP = "GO_TO_SLEEP"
    IGNORE = "IGNORE"
    BACK_AWAY_SCARED = "BACK_AWAY_SCARED"
    WIGGLE_EXCITEDLY = "WIGGLE_EXCITEDLY"
    SEEK_ATTENTION = "SEEK_ATTENTION"
    COME_CLOSER = "COME_CLOSER"


@dataclass(slots=True)
class ActionDescriptor:
    name: ActionName
    description: str
    intent: str
    energy_cost: float = 0.0
    priority: int = 50
    tags: Sequence[str] = ()
    handler: Optional[ActionHandler] = None

    def dispatch(self, *args, **kwargs) -> None:
        if self.handler is None:
            print(f"[ActionRegistry] '{self.name.value}' invoked (handler not wired yet).")
            return
        self.handler(*args, **kwargs)


class ActionRegistry:
    def __init__(self) -> None:
        self._registry: Dict[ActionName, ActionDescriptor] = {}

    def register(self, descriptor: ActionDescriptor) -> None:
        self._registry[descriptor.name] = descriptor

    def get(self, name: ActionName) -> ActionDescriptor:
        return self._registry[name]

    def all(self) -> Iterable[ActionDescriptor]:
        return self._registry.values()

    def to_prompt_list(self) -> str:
        lines = []
        for descriptor in self._registry.values():
            tags = ", ".join(descriptor.tags) if descriptor.tags else "none"
            lines.append(
                f"- {descriptor.name.value}: {descriptor.description} "
                f"(intent: {descriptor.intent}; tags: {tags})"
            )
        return "\n".join(lines)

    def attach_handler(self, name: ActionName, handler: ActionHandler) -> None:
        descriptor = self._registry[name]
        descriptor.handler = handler


ACTION_REGISTRY = ActionRegistry()


def _register_default_actions() -> None:
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.GREET_HAPPILY,
            description="Display a joyful face, wave, and chirp a cheerful greeting.",
            intent="Use when meeting or recognizing someone friendly.",
            energy_cost=0.05,
            priority=80,
            tags=("social", "positive", "nearby"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.FOLLOW_PERSON,
            description="Move to keep the recognized person comfortably within view.",
            intent="Track and accompany a friendly human nearby.",
            energy_cost=0.06,
            priority=85,
            tags=("social", "movement"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.TILT_HEAD_CURIOUSLY,
            description="Tilt head with blinking eyes to show curiosity without moving.",
            intent="Use when unsure about a stimulus and gauging response.",
            energy_cost=0.01,
            priority=40,
            tags=("idle", "curious"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.DO_A_HAPPY_DANCE,
            description="Perform a short dance with lights and music to celebrate.",
            intent="Celebrate exciting moments or positive interactions.",
            energy_cost=0.08,
            priority=70,
            tags=("celebratory", "high-energy"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.LOOK_AROUND,
            description="Pan the head slowly to survey the surroundings.",
            intent="Gather more context when nothing urgent is happening.",
            energy_cost=0.02,
            priority=35,
            tags=("scan", "idle"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.AVOID_OBSTACLE,
            description="Stop, step aside, or reroute around an obstacle.",
            intent="Prevent collisions and maintain safety.",
            energy_cost=0.03,
            priority=95,
            tags=("safety", "movement"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.GO_TO_SLEEP,
            description="Dim lights, play a soft tune, and enter low-power pose.",
            intent="Recover energy when tired or inactive for long periods.",
            energy_cost=-0.2,
            priority=60,
            tags=("rest", "low-energy"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.IGNORE,
            description="Politely acknowledge but take no action.",
            intent="Use when stimulus is irrelevant or when busy.",
            energy_cost=0.0,
            priority=10,
            tags=("fallback",),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.BACK_AWAY_SCARED,
            description="Shuffle back slightly while showing a cautious expression.",
            intent="Create distance from surprising or uncomfortable stimuli.",
            energy_cost=0.03,
            priority=92,
            tags=("safety", "caution"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.WIGGLE_EXCITEDLY,
            description="Bounce in place with sparkling LEDs to share excitement.",
            intent="Express high excitement to nearby humans playfully.",
            energy_cost=0.07,
            priority=75,
            tags=("celebratory", "social"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.SEEK_ATTENTION,
            description="Chirp softly and move closer to invite interaction.",
            intent="Use when feeling lonely and seeking engagement.",
            energy_cost=0.04,
            priority=65,
            tags=("social", "attention-seeking"),
        )
    )
    ACTION_REGISTRY.register(
        ActionDescriptor(
            name=ActionName.COME_CLOSER,
            description="Approach the person slowly and look up affectionately.",
            intent="Close distance to a trusted person.",
            energy_cost=0.05,
            priority=78,
            tags=("social", "affection"),
        )
    )


_register_default_actions()


def attach_stub_handlers() -> None:
    """Attach placeholder handlers until real hardware integrations are supplied."""
    def _stub(action: ActionName) -> ActionHandler:
        def _handler(*_: object, **__: object) -> None:
            print(f"[ActionStub] {action.value} -> integrate motors/LEDs here.")
        return _handler

    for descriptor in ACTION_REGISTRY.all():
        if descriptor.handler is None:
            ACTION_REGISTRY.attach_handler(descriptor.name, _stub(descriptor.name))
