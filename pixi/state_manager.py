from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

from pixi.actions import ActionName


class Mood(str, Enum):
    """High-level emotional states that influence Pixi's personality."""

    CURIOUS = "curious"
    HAPPY = "happy"
    PLAYFUL = "playful"
    ALERT = "alert"
    SLEEPY = "sleepy"
    SCARED = "scared"
    EXCITED = "excited"
    LONELY = "lonely"


@dataclass(slots=True)
class InternalState:
    """Structured container for Pixi's internal signals."""

    mood: Mood = Mood.CURIOUS
    energy: float = 0.85  # 0.0 - 1.0 scale
    curiosity: float = 0.65  # 0.0 - 1.0 scale
    confidence: float = 0.55  # How sure Pixi feels about its surroundings
    recognized_person: Optional[str] = None
    last_action: Optional[ActionName] = None
    attention_hunger: float = 0.35
    excitement: float = 0.45
    caution: float = 0.3


class StateManager:
    """Tracks and updates Pixi's feelings, energy, and memories."""

    def __init__(
        self,
        *,
        boredom_timeout: float = 15.0,
        energy_decay_per_second: float = 0.003,
        curiosity_rise_per_second: float = 0.004,
        max_recent_actions: int = 8,
    ) -> None:
        self._state = InternalState()
        self._last_interaction_ts = time.time()
        self._last_tick_ts = self._last_interaction_ts
        self._boredom_timeout = boredom_timeout
        self._energy_decay = energy_decay_per_second
        self._curiosity_rise = curiosity_rise_per_second
        self._recent_actions: Deque[ActionName] = deque(maxlen=max_recent_actions)

    # ------------------------------------------------------------------
    # Timers & progression
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    def tick(self) -> None:
        """Updates energy/curiosity deltas based on elapsed real time."""
        now = time.time()
        delta = max(0.0, now - self._last_tick_ts)
        self._last_tick_ts = now

        # Energy decreases slowly unless Pixi recently rested.
        new_energy = self._state.energy - (self._energy_decay * delta)
        self._state.energy = self._clamp(new_energy, 0.1, 1.0)

        # Curiosity increases when Pixi is bored, and dips after recent activity.
        boredom = self.time_since_last_interaction() 
        curiosity_delta = self._curiosity_rise * delta
        if boredom > self._boredom_timeout:
            curiosity_delta *= 1.8
        if self._recent_actions and self._recent_actions[-1] == ActionName.FOLLOW_PERSON:
            curiosity_delta *= 0.4  # Following someone satisfies curiosity for a moment.

        self._state.curiosity = self._clamp(self._state.curiosity + curiosity_delta, 0.1, 1.0)

        attention_gain = 0.015 * delta
        if boredom > self._boredom_timeout:
            attention_gain *= 2.0
        self._state.attention_hunger = self._clamp(self._state.attention_hunger + attention_gain, 0.0, 1.0)

        excite_target = 0.4
        excite_delta = (excite_target - self._state.excitement) * 0.35 * delta
        self._state.excitement = self._clamp(self._state.excitement + excite_delta, 0.1, 1.0)

        caution_drop = 0.025 * delta
        self._state.caution = self._clamp(self._state.caution - caution_drop, 0.1, 1.0)

    def time_since_last_interaction(self) -> float:
        return time.time() - self._last_interaction_ts

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------
    def register_interaction(self, person: Optional[str] = None) -> None:
        """Records that Pixi interacted with someone."""
        self._last_interaction_ts = time.time()
        self._state.curiosity = self._clamp(self._state.curiosity - 0.1, 0.2, 1.0)
        self._state.confidence = self._clamp(self._state.confidence + 0.05, 0.0, 1.0)
        self._state.attention_hunger = self._clamp(self._state.attention_hunger - 0.25, 0.0, 1.0)
        self._state.excitement = self._clamp(self._state.excitement + 0.04, 0.1, 1.0)
        self._state.caution = self._clamp(self._state.caution - 0.05, 0.1, 1.0)
        if person:
            self._state.recognized_person = person

    def update_face_target(
        self,
        *,
        face_id: str,
        center_x: float,
        center_y: float,
        area: float,
        confidence: float,
    ) -> None:
        """Lightweight update when a face stays in view to aid tracking behaviour."""

        self._state.recognized_person = face_id
        self._last_interaction_ts = time.time()

        # Seeing a confident face lowers caution slightly and builds confidence over time.
        confidence_delta = 0.02 * self._clamp(confidence, 0.0, 1.0)
        self._state.confidence = self._clamp(self._state.confidence + confidence_delta, 0.0, 1.0)
        self._state.caution = self._clamp(self._state.caution - (confidence_delta * 0.6), 0.1, 1.0)

        # When someone is close-by (large area), satiate attention hunger a little.
        if area > 0.02:
            self._state.attention_hunger = self._clamp(self._state.attention_hunger - 0.01, 0.0, 1.0)

    def update_mood(self, new_mood: Mood) -> None:
        if self._state.mood != new_mood:
            print(f"[State] Mood changed from {self._state.mood.value} to {new_mood.value}")
            self._state.mood = new_mood

    def update_after_action(self, action: ActionName) -> None:
        """Adjusts internal stats depending on the chosen action."""
        self._recent_actions.append(action)
        self._state.last_action = action

        if action == ActionName.GO_TO_SLEEP:
            self._state.energy = self._clamp(self._state.energy + 0.25, 0.1, 1.0)
            self._state.excitement = self._clamp(self._state.excitement - 0.1, 0.1, 1.0)
            self._state.attention_hunger = self._clamp(self._state.attention_hunger - 0.2, 0.0, 1.0)
            self.update_mood(Mood.SLEEPY)
        elif action == ActionName.DO_A_HAPPY_DANCE:
            self._state.energy = self._clamp(self._state.energy - 0.07, 0.2, 1.0)
            self._state.excitement = self._clamp(self._state.excitement + 0.2, 0.1, 1.0)
            self.update_mood(Mood.PLAYFUL)
        elif action == ActionName.WIGGLE_EXCITEDLY:
            self._state.energy = self._clamp(self._state.energy - 0.05, 0.2, 1.0)
            self._state.excitement = self._clamp(self._state.excitement + 0.25, 0.1, 1.0)
            self.update_mood(Mood.EXCITED)
        elif action == ActionName.FOLLOW_PERSON:
            self._state.energy = self._clamp(self._state.energy - 0.04, 0.2, 1.0)
            self._state.confidence = self._clamp(self._state.confidence + 0.08, 0.0, 1.0)
            self._state.attention_hunger = self._clamp(self._state.attention_hunger - 0.15, 0.0, 1.0)
            self.update_mood(Mood.CURIOUS)
        elif action == ActionName.GREET_HAPPILY:
            self._state.confidence = self._clamp(self._state.confidence + 0.05, 0.0, 1.0)
            self._state.excitement = self._clamp(self._state.excitement + 0.1, 0.1, 1.0)
            self.update_mood(Mood.HAPPY)
        elif action == ActionName.SEEK_ATTENTION:
            self._state.attention_hunger = self._clamp(self._state.attention_hunger - 0.3, 0.0, 1.0)
            self._state.curiosity = self._clamp(self._state.curiosity + 0.05, 0.1, 1.0)
            self.update_mood(Mood.LONELY)
        elif action == ActionName.COME_CLOSER:
            self._state.attention_hunger = self._clamp(self._state.attention_hunger - 0.2, 0.0, 1.0)
            self._state.confidence = self._clamp(self._state.confidence + 0.06, 0.0, 1.0)
            self._state.caution = self._clamp(self._state.caution - 0.08, 0.1, 1.0)
            self.update_mood(Mood.CURIOUS)
        elif action == ActionName.BACK_AWAY_SCARED:
            self._state.caution = self._clamp(self._state.caution + 0.25, 0.1, 1.0)
            self._state.confidence = self._clamp(self._state.confidence - 0.1, 0.0, 1.0)
            self.update_mood(Mood.SCARED)
        elif action == ActionName.AVOID_OBSTACLE:
            self._state.caution = self._clamp(self._state.caution + 0.1, 0.1, 1.0)
            self.update_mood(Mood.ALERT)
        elif action == ActionName.LOOK_AROUND:
            self._state.curiosity = self._clamp(self._state.curiosity - 0.02, 0.1, 1.0)
        elif action == ActionName.TILT_HEAD_CURIOUSLY:
            self._state.curiosity = self._clamp(self._state.curiosity + 0.01, 0.1, 1.0)
        elif action == ActionName.IGNORE:
            self._state.attention_hunger = self._clamp(self._state.attention_hunger + 0.05, 0.0, 1.0)

        if action not in {ActionName.IGNORE, ActionName.BACK_AWAY_SCARED, ActionName.AVOID_OBSTACLE}:
            self.register_interaction(self._state.recognized_person)

    # ------------------------------------------------------------------
    # State exposure
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Returns a serialisable snapshot for prompts or telemetry."""
        last_action_value = self._state.last_action.value if self._state.last_action else "NONE"
        return {
            "mood": self._state.mood.value,
            "energy": round(self._state.energy, 2),
            "curiosity": round(self._state.curiosity, 2),
            "confidence": round(self._state.confidence, 2),
            "attention_hunger": round(self._state.attention_hunger, 2),
            "excitement": round(self._state.excitement, 2),
            "caution": round(self._state.caution, 2),
            "time_since_last_interaction": round(self.time_since_last_interaction(), 1),
            "recognized_person": self._state.recognized_person or "unknown",
            "last_action": last_action_value,
            "recent_actions": [action.value for action in self._recent_actions],
        }

