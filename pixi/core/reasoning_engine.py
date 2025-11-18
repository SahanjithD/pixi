from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from pixi.core.actions import ACTION_REGISTRY, ActionName
from pixi.core.state_manager import StateManager

PROMPT_TEMPLATE = (
    "You are Pixi's cognition module, responsible for choosing exactly one high-level action.\n"
    "Respect Pixi's core personality: curious, adorable, cautious around surprises, playful when energized.\n"
    "Base all decisions on state metrics, sensor events, and the action catalogue provided.\n\n"
    "Current internal state:\n{state_block}\n\n"
    "Recent events:\n{event_block}\n\n"
    "Action catalogue:\n{actions_block}\n\n"
    "Guidelines:\n"
    "- When caution > 0.7, prefer AVOID_OBSTACLE or BACK_AWAY_SCARED before playful actions.\n"
    "- When attention_hunger > 0.65, consider SEEK_ATTENTION or COME_CLOSER if safe.\n"
    "- When excitement > 0.75, favour DO_A_HAPPY_DANCE or WIGGLE_EXCITEDLY to express energy.\n"
    "- When energy < 0.25, consider GO_TO_SLEEP unless there is an urgent event.\n"
    "- Only return actions from the catalogue. Vary choices to avoid repetition.\n\n"
    "Respond with valid JSON: {{\"action\": \"<ACTION_NAME>\", \"reason\": \"<brief rationale>\"}}."
)


class ReasoningEngine:
    def __init__(
        self,
        *,
        state_manager: Optional[StateManager] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.4,
    ) -> None:
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("Missing GROQ_API_KEY in environment.")

        self._state_manager = state_manager or StateManager()
        self._actions = ACTION_REGISTRY
        model = model_name or os.getenv("GROQ_MODEL_NAME") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        self._llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=temperature,
        )
        self._chain = self._llm

    def decide_action(self, events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the LangChain reasoning loop and return the chosen action."""
        self._state_manager.tick()
        state_snapshot = self._state_manager.get_state()

        payload = {
            "state_block": self._format_state(state_snapshot),
            "event_block": self._format_events(events),
            "actions_block": self._actions.to_prompt_list(),
        }

        prompt_text = PROMPT_TEMPLATE.format(**payload)
        result = self._chain.invoke(prompt_text)
        if hasattr(result, "content"):
            raw_text = result.content
        else:
            raw_text = str(result)

        action, reason = self._extract_action(raw_text)
        self._state_manager.update_after_action(action)

        return {
            "action": action,
            "reason": reason,
            "raw_response": raw_text,
            "state": state_snapshot,
        }

    def _format_state(self, state: Dict[str, Any]) -> str:
        keys = [
            "mood",
            "energy",
            "curiosity",
            "confidence",
            "attention_hunger",
            "excitement",
            "caution",
            "time_since_last_interaction",
            "recognized_person",
            "last_action",
        ]
        return "\n".join(f"- {key}: {state.get(key, 'n/a')}" for key in keys)

    def _format_events(self, events: Iterable[Dict[str, Any]]) -> str:
        items: List[str] = []
        for event in events:
            summary = event.get("summary") or event.get("type", "unknown_event")
            weight = event.get("weight", "n/a")
            items.append(f"- {summary} (weight={weight})")
        return "\n".join(items) if items else "- none"

    def _extract_action(self, raw_text: str) -> tuple[ActionName, str]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = {}

        action_raw = str(payload.get("action", "")).strip().upper()
        reason = str(payload.get("reason", "")).strip() or "No reason provided."

        action = self._normalise_action(action_raw)
        return action, reason

    def _normalise_action(self, action_raw: str) -> ActionName:
        for candidate in ActionName:
            if action_raw in {candidate.value.upper(), candidate.name}:
                return candidate

        synonyms = {
            "HAPPY_DANCE": ActionName.DO_A_HAPPY_DANCE,
            "TILT_HEAD": ActionName.TILT_HEAD_CURIOUSLY,
            "FOLLOW": ActionName.FOLLOW_PERSON,
            "SLEEP": ActionName.GO_TO_SLEEP,
            "BACK_AWAY": ActionName.BACK_AWAY_SCARED,
            "WIGGLE": ActionName.WIGGLE_EXCITEDLY,
            "ATTENTION": ActionName.SEEK_ATTENTION,
            "CLOSER": ActionName.COME_CLOSER,
        }
        for key, mapped in synonyms.items():
            if key in action_raw:
                return mapped

        return ActionName.LOOK_AROUND


if __name__ == "__main__":
    engine = ReasoningEngine()
    demo_events = [
        {"summary": "Person waves at Pixi", "weight": 0.7},
        {"summary": "No obstacles detected", "weight": 0.1},
    ]
    decision = engine.decide_action(demo_events)
    print(decision)
