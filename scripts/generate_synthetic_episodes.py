from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import struct
import zlib


WIDTH = 640
HEIGHT = 360

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["11111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}

SCENES = [
    ("LIBRARY", "red", (158, 48, 42), "library entrance"),
    ("PAVILION", "brown", (150, 88, 48), "wooden pavilion"),
    ("LAB", "blue", (58, 92, 155), "laboratory door"),
    ("GYM", "green", (56, 130, 78), "gym entrance"),
    ("ADMIN", "white", (210, 210, 195), "administration entrance"),
    ("CANTEEN", "orange", (188, 112, 50), "canteen doorway"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/episodes/synthetic_30")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31121)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest_rows = []
    for episode_index in range(args.episodes):
        scene = SCENES[episode_index % len(SCENES)]
        route_variant = episode_index % 5
        goal_distance = 18.0 + 0.8 * episode_index
        lateral_goal = ((episode_index % 7) - 3) * 0.35
        instruction = make_instruction(scene, route_variant)
        cue_drop_start = max(2, args.steps // 3)
        cue_drop_end = min(args.steps - 2, cue_drop_start + 2 + (episode_index % 3))

        for step in range(args.steps):
            progress = step / max(args.steps - 1, 1)
            cue_visible = not (cue_drop_start <= step <= cue_drop_end)
            image_name = f"episode_{episode_index + 1:03d}_step_{step:02d}.png"
            image_path = image_dir / image_name
            draw_scene(image_path, scene, progress, cue_visible, route_variant, rng)
            pose_x = progress * goal_distance
            pose_y = math.sin(progress * math.pi) * 0.4 * ((episode_index % 3) - 1)
            manifest_rows.append(
                {
                    "episode_id": f"synthetic_{episode_index + 1:03d}",
                    "image_path": str(image_path).replace("\\", "/"),
                    "instruction": instruction,
                    "pose": {"x": round(pose_x, 3), "y": round(pose_y, 3), "z": 0.0, "yaw": 0.0},
                    "camera": {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "fx": 550.0,
                        "fy": 550.0,
                        "cx": WIDTH / 2.0,
                        "cy": HEIGHT / 2.0,
                    },
                    "goal_position": {
                        "x": round(goal_distance, 3),
                        "y": round(lateral_goal, 3),
                        "z": 0.0,
                    },
                    "shortest_path_m": round(goal_distance, 3),
                    "cue_available_gt": cue_visible,
                }
            )

    write_jsonl(output_root / "manifest.jsonl", manifest_rows)
    write_dataset_readme(output_root, args.episodes, args.steps, args.seed)
    print(f"Wrote {len(manifest_rows)} steps for {args.episodes} episodes to {output_root}")


def make_instruction(scene: tuple[str, str, tuple[int, int, int], str], route_variant: int) -> str:
    _label, color_name, _color, target = scene
    route_bits = [
        "Follow the paved path and avoid the grass.",
        "Stay on the central walkway and keep off the lawn.",
        "Move between the trees, then continue on the path.",
        "Pass the open lawn without stepping onto it.",
        "Keep the building centered and use the safe path.",
    ]
    return f"Find the {color_name} {target}. {route_bits[route_variant]}"


def draw_scene(
    path: Path,
    scene: tuple[str, str, tuple[int, int, int], str],
    progress: float,
    cue_visible: bool,
    route_variant: int,
    rng: random.Random,
) -> None:
    label, _color_name, building_color, _target = scene
    canvas = [[(0, 0, 0) for _ in range(WIDTH)] for __ in range(HEIGHT)]

    for y in range(HEIGHT):
        for x in range(WIDTH):
            if y < 105:
                canvas[y][x] = (118, 180, 235)
            elif y < 225:
                canvas[y][x] = (72, 148, 70)
            else:
                left = int(145 - progress * 25 + (y - 225) * -0.45)
                right = int(495 + progress * 25 + (y - 225) * 0.45)
                if left < x < right:
                    shade = ((x // 22) + (y // 18) + route_variant) % 2
                    canvas[y][x] = (190, 182, 164) if shade else (214, 205, 186)
                else:
                    canvas[y][x] = (48, 140, 58)

    draw_circle(canvas, 75, 160, 70, (28, 112, 45))
    draw_circle(canvas, 565, 160, 70, (28, 112, 45))
    draw_rect(canvas, 145, 105, 495, 225, building_color)
    add_bricks(canvas, 145, 105, 495, 225, building_color)

    draw_rect(canvas, 255, 140, 385, 225, (214, 205, 188))
    draw_rect(canvas, 285, 165, 355, 225, (42, 50, 58))
    draw_rect(canvas, 185, 142, 245, 198, (34, 62, 75))
    draw_rect(canvas, 395, 142, 455, 198, (34, 62, 75))

    sign_color = (150, 36, 34) if label == "LIBRARY" else (50, 70, 92)
    draw_rect(canvas, 230, 112, 410, 140, sign_color)
    draw_text(canvas, label, 244, 117, 3, (245, 245, 230))

    if not cue_visible:
        # Cover or shift the semantic cue to simulate cue interruption.
        if route_variant % 2 == 0:
            draw_rect(canvas, 220, 102, 420, 205, (35, 118, 52))
        else:
            draw_rect(canvas, 0, 90, 240, 230, (35, 118, 52))
            draw_rect(canvas, 500, 90, WIDTH, 230, (35, 118, 52))

    # Deterministic small texture on grass.
    for _ in range(55):
        x = rng.randrange(0, WIDTH)
        y = rng.randrange(230, HEIGHT)
        if canvas[y][x] == (48, 140, 58):
            draw_rect(canvas, x, y, min(x + 2, WIDTH - 1), min(y + 2, HEIGHT - 1), (60, 160, 65))

    write_png(path, canvas)


def draw_rect(canvas: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            canvas[y][x] = color


def draw_circle(canvas: list[list[tuple[int, int, int]]], cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    r2 = radius * radius
    for y in range(max(0, cy - radius), min(HEIGHT, cy + radius)):
        for x in range(max(0, cx - radius), min(WIDTH, cx + radius)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                canvas[y][x] = color


def add_bricks(
    canvas: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    base: tuple[int, int, int],
) -> None:
    for y in range(y0, y1, 9):
        for x in range(x0, x1):
            canvas[y][x] = tuple(max(0, c - 28) for c in base)
    for y in range(y0, y1):
        offset = 12 if ((y - y0) // 9) % 2 else 0
        for x in range(x0 + offset, x1, 24):
            canvas[y][x] = tuple(max(0, c - 22) for c in base)


def draw_text(
    canvas: list[list[tuple[int, int, int]]],
    text: str,
    x: int,
    y: int,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, value in enumerate(row):
                if value == "1":
                    draw_rect(
                        canvas,
                        cursor + gx * scale,
                        y + gy * scale,
                        cursor + (gx + 1) * scale,
                        y + (gy + 1) * scale,
                        color,
                    )
        cursor += 6 * scale


def write_png(path: Path, canvas: list[list[tuple[int, int, int]]]) -> None:
    raw = bytearray()
    for row_pixels in canvas:
        raw.append(0)
        for r, g, b in row_pixels:
            raw.extend([r, g, b])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_dataset_readme(output_root: Path, episodes: int, steps: int, seed: int) -> None:
    (output_root / "README.md").write_text(
        f"""# Synthetic 30-Episode VLN Dataset

This is a small synthetic dataset for engineering validation of the TARIC/VLN pipeline.

- Episodes: {episodes}
- Steps per episode: {steps}
- Total manifest rows: {episodes * steps}
- Image size: {WIDTH}x{HEIGHT}
- Generator seed: {seed}

The scenes contain simplified outdoor paths, grass, buildings, target signs, and cue-interruption steps. This dataset is for smoke testing and pipeline validation only; it is not a paper-quality benchmark.

Run:

```bash
python scripts/run_offline_episode.py \\
  --manifest data/episodes/synthetic_30/manifest.jsonl \\
  --output outputs/synthetic_30_run.jsonl \\
  --qwen

python scripts/summarize_run.py --input outputs/synthetic_30_run.jsonl
```
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
