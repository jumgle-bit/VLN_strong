from __future__ import annotations

import math

from taric_vln.config import TaricConfig
from taric_vln.geometry import sector_centers
from taric_vln.grounding import TraversabilityGrounder


def test_grounder_snaps_to_nearest_traversable_sector() -> None:
    config = TaricConfig(heading_sectors=5, max_bearing_deg=60.0)
    grounder = TraversabilityGrounder(config)
    centers = sector_centers(config)
    scores = [0.2, 0.2, 0.1, 0.9, 0.2]

    result = grounder.ground(0.0, scores)

    assert result.status == "snapped_to_traversable"
    assert result.selected_sector == 3
    assert math.isclose(result.executable_bearing_rad, centers[3])


def test_grounder_keeps_preferred_when_sector_is_traversable() -> None:
    config = TaricConfig(heading_sectors=5, max_bearing_deg=60.0)
    grounder = TraversabilityGrounder(config)
    result = grounder.ground(0.02, [0.2, 0.2, 0.9, 0.2, 0.2])

    assert result.status == "kept_preferred"
    assert math.isclose(result.executable_bearing_rad, 0.02)
