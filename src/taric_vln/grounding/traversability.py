from __future__ import annotations

from dataclasses import dataclass

from taric_vln.config import TaricConfig
from taric_vln.geometry import angle_diff, sector_centers


@dataclass(frozen=True)
class GroundingResult:
    executable_bearing_rad: float
    selected_sector: int | None
    status: str
    traversable_indices: list[int]


class TraversabilityGrounder:
    def __init__(self, config: TaricConfig, platform: str = "scout") -> None:
        self.config = config
        self.platform = platform

    @property
    def threshold(self) -> float:
        if self.platform.lower() in {"go1", "legged", "quadruped"}:
            return self.config.go1_traversability_threshold
        return self.config.scout_traversability_threshold

    def ground(
        self,
        preferred_bearing_rad: float,
        traversability_scores: list[float],
        sigmas: list[float] | None = None,
        threshold: float | None = None,
        cone_half_angle_rad: float | None = None,
    ) -> GroundingResult:
        centers = sector_centers(self.config)
        scores = self._fit_scores(traversability_scores)
        sigma_values = self._fit_scores(sigmas or [0.0] * len(scores), fill=0.0)
        threshold_value = self.threshold if threshold is None else threshold

        robust_scores = [
            score - self.config.risk_lambda * sigma
            for score, sigma in zip(scores, sigma_values)
        ]
        candidates = [
            i
            for i, score in enumerate(robust_scores)
            if score >= threshold_value
            and (
                cone_half_angle_rad is None
                or angle_diff(centers[i], preferred_bearing_rad) <= cone_half_angle_rad
            )
        ]

        if not candidates:
            return GroundingResult(
                executable_bearing_rad=preferred_bearing_rad,
                selected_sector=None,
                status="no_traversable_sector",
                traversable_indices=[],
            )

        preferred_sector = min(
            range(len(centers)), key=lambda i: angle_diff(centers[i], preferred_bearing_rad)
        )
        if preferred_sector in candidates:
            return GroundingResult(
                executable_bearing_rad=preferred_bearing_rad,
                selected_sector=preferred_sector,
                status="kept_preferred",
                traversable_indices=candidates,
            )

        selected = min(candidates, key=lambda i: angle_diff(centers[i], preferred_bearing_rad))
        return GroundingResult(
            executable_bearing_rad=centers[selected],
            selected_sector=selected,
            status="snapped_to_traversable",
            traversable_indices=candidates,
        )

    def _fit_scores(self, scores: list[float], fill: float = 0.0) -> list[float]:
        n = self.config.heading_sectors
        if not scores:
            return [fill] * n
        fitted = [float(x) for x in scores[:n]]
        if len(fitted) < n:
            fitted.extend([fitted[-1] if fitted else fill] * (n - len(fitted)))
        return fitted
