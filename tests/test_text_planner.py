from __future__ import annotations

from taric_vln.perception.text_planner import instruction_plan_from_payload


def test_instruction_plan_from_payload() -> None:
    plan = instruction_plan_from_payload(
        "Find the red library entrance.",
        {
            "exploration_phrase": "follow the path",
            "goal_phrase": "red library entrance",
            "avoid_phrases": ["grass"],
            "route_cues": ["path", "library"],
        },
    )

    assert plan.goal_phrase == "red library entrance"
    assert plan.avoid_phrases == ["grass"]
