"""
Draw bounding boxes from nid_fields.json on the NID image.
Usage: python draw_nid_boxes.py [image_path] [json_path]
Defaults: IMAGE_PATH from nid_ocr.py or "arafat.jpg", nid_fields.json in cwd.
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Paths (override with script args: python draw_nid_boxes.py <image> [json])
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR / "shadid_front_nid.png"
NID_JSON_PATH = SCRIPT_DIR / "nid_fields.json"

# Colors (R, G, B) per field - distinct colors
FIELD_COLORS = {
    "english_name": (255, 100, 100),   # red
    "bangla_name": (100, 200, 255),    # blue
    "father_name": (100, 255, 150),    # green
    "mother_name": (255, 200, 100),   # orange
    "date_of_birth": (200, 100, 255), # purple
    "id_no": (100, 255, 255),         # cyan
}
DEFAULT_COLOR = (180, 180, 180)
BOX_THICKNESS = 3
LABEL_HEIGHT = 22


def get_font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except OSError:
            return ImageFont.load_default()


def draw_boxes_on_image(
    image_path: Path, json_path: Path, output_path: Path, crops_dir: Path | None = None
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_draw = img.copy()  # draw on copy so crops stay clean
    draw = ImageDraw.Draw(img_draw)
    font = get_font(14)
    w, h = img.size

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for field_name, item in data.items():
        if item is None:
            continue
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x_min, y_min, x_max, y_max = [int(x) for x in bbox]
        # Clamp to image bounds
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(x_min + 1, min(x_max, w))
        y_max = max(y_min + 1, min(y_max, h))
        color = FIELD_COLORS.get(field_name, DEFAULT_COLOR)

        # Crop and save from original image (no drawings)
        if crops_dir is not None:
            cropped = img.crop((x_min, y_min, x_max, y_max))
            crop_path = crops_dir / f"{field_name}.png"
            cropped.save(crop_path)
            print(f"Cropped: {crop_path}")

        # Draw rectangle on output image
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=color,
            width=BOX_THICKNESS,
        )
        label = field_name.replace("_", " ").title()
        label_y = max(2, y_min - LABEL_HEIGHT)
        draw.rectangle(
            [(x_min, label_y), (x_max, y_min)],
            fill=color,
            outline=color,
        )
        draw.text((x_min + 4, label_y + 2), label, fill=(0, 0, 0), font=font)

    img_draw.save(output_path)
    print(f"Saved: {output_path}")
    try:
        img_draw.show()
    except Exception:
        pass  # no display


def main():
    if len(sys.argv) >= 2:
        image_path = Path(sys.argv[1])
    else:
        image_path = IMAGE_PATH
    if len(sys.argv) >= 3:
        json_path = Path(sys.argv[2])
    else:
        json_path = NID_JSON_PATH

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    if not json_path.exists():
        print(f"Error: JSON not found: {json_path}")
        sys.exit(1)

    output_path = image_path.parent / f"{image_path.stem}_boxes.png"
    crops_dir = image_path.parent / f"{image_path.stem}_crops"
    draw_boxes_on_image(image_path, json_path, output_path, crops_dir=crops_dir)


if __name__ == "__main__":
    main()
