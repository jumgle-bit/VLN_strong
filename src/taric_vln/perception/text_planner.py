from __future__ import annotations

import json
from typing import Any

from taric_vln.config import TaricConfig
from taric_vln.perception.deepseek_client import DeepSeekVLMClient, extract_json_object
from taric_vln.types import InstructionPlan


class DeepSeekTextPlanner:
    """Use DeepSeek's text API for instruction decomposition only."""

    def __init__(self, client: DeepSeekVLMClient | None = None) -> None:
        self.client = client or DeepSeekVLMClient(cache_dir=".taric_cache/deepseek_text")

    def plan(
        self,
        instruction: str,
        config: TaricConfig | None = None,
    ) -> InstructionPlan:
        config = config or TaricConfig()
        try:
            response = self.client._post_json(self._payload(instruction, config))
            content = response["choices"][0]["message"]["content"]
            payload = extract_json_object(content)
            payload["_api_meta"] = {
                "model": self.client.model,
                "usage": response.get("usage", {}),
            }
            return instruction_plan_from_payload(instruction, payload)
        except Exception as exc:
            return InstructionPlan(
                instruction=instruction,
                exploration_phrase=instruction,
                goal_phrase=instruction,
                error=str(exc),
            )

    def _payload(self, instruction: str, config: TaricConfig) -> dict[str, Any]:
        schema = {
            "exploration_phrase": "coarse route or exploration phrase",
            "goal_phrase": "specific destination or target phrase",
            "avoid_phrases": ["terrain or objects to avoid"],
            "route_cues": ["ordered route landmarks or path hints"],
        }
        return {
            "model": self.client.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You decompose outdoor VLN instructions for a robot. "
                        "Return only strict JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Instruction: {instruction}\n"
                        f"Heading sectors available downstream: {config.heading_sectors}\n"
                        "Extract a concise goal phrase, a broader exploration phrase, "
                        "avoid constraints, and route cues. Return JSON matching:\n"
                        f"{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
            ],
        }


def instruction_plan_from_payload(instruction: str, payload: dict[str, Any]) -> InstructionPlan:
    return InstructionPlan(
        instruction=instruction,
        exploration_phrase=str(payload.get("exploration_phrase", instruction)),
        goal_phrase=str(payload.get("goal_phrase", instruction)),
        avoid_phrases=parse_string_list(payload.get("avoid_phrases")),
        route_cues=parse_string_list(payload.get("route_cues")),
        raw=payload,
    )


def parse_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []
