import json
import re
import sys
from pathlib import Path
# bam
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

# Ensure UTF-8 output for Bengali/Unicode in terminal
try:
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass  # reconfigure not available or not a TTY

IMAGE_PATH = "preprocessed_samples/shourov_nid_front.png"
OUTPUT_TXT = "ocr_result.txt"
OUTPUT_JSON = "nid_fields.json"

# Label patterns for NID card (English / Bangla)
NAME_PREFIX = re.compile(r"^Name:\s*", re.I)
DOB_PREFIX = re.compile(r"Date\s+of\s+Birth:\s*", re.I)
IDNO_PREFIX = re.compile(r"ID\s+NO:\s*", re.I)
BANGLA_NAME_LABEL = "নাম:"
BANGLA_FATHER = "পিতা:"
BANGLA_MOTHER = "মাতা:"


def has_bengali(s: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", s))


def is_header_line(s: str) -> bool:
    lower = s.lower()
    return (
        "government" in lower
        or "republic" in lower
        or "national id" in lower
        or "জাতীয়" in s
        or "গণপ্রজাতন্ত্রী" in s
    )


def extract_nid_fields(lines):
    """Extract candidate english_name, bangla_name, father, mother, date_of_birth, id_no with bboxes."""
    # Each line has .text and .bbox = [x_min, y_min, x_max, y_max]
    out = {
        "english_name": None,
        "bangla_name": None,
        "father_name": None,
        "mother_name": None,
        "date_of_birth": None,
        "id_no": None,
    }

    for i, line in enumerate(lines):
        text = (line.text or "").strip()
        bbox = list(line.bbox)  # [x_min, y_min, x_max, y_max]

        # English name: "Name: YASIR ARAFAT PRODHAN"
        if NAME_PREFIX.search(text):
            value = NAME_PREFIX.sub("", text).strip()
            out["english_name"] = {"bbox": bbox, "text": value}

        # Date of Birth
        if DOB_PREFIX.search(text):
            value = DOB_PREFIX.sub("", text).strip()
            out["date_of_birth"] = {"bbox": bbox, "text": value}

        # ID NO
        if IDNO_PREFIX.search(text):
            value = IDNO_PREFIX.sub("", text).strip()
            out["id_no"] = {"bbox": bbox, "text": value}

        # Bangla labels: value can be on same line, or next line (father), or *previous* line (mother)
        if BANGLA_FATHER in text:
            value = text.split(BANGLA_FATHER, 1)[-1].strip() or (
                lines[i + 1].text.strip() if i + 1 < len(lines) else ""
            )
            out["father_name"] = {"bbox": bbox, "text": value}

        if BANGLA_MOTHER in text:
            after_label = text.split(BANGLA_MOTHER, 1)[-1].strip()
            if after_label:
                value = after_label
            elif i > 0 and (lines[i - 1].text or "").strip():
                # NID layout: mother name often appears on the line *before* মাতা:
                prev_text = (lines[i - 1].text or "").strip()
                value = prev_text
                bbox = list(lines[i - 1].bbox)
            else:
                value = (lines[i + 1].text.strip() if i + 1 < len(lines) else "")
            out["mother_name"] = {"bbox": bbox, "text": value}

        # Bangla name: can be on line *before* নাম: or line *after* নাম: (NID layout varies)
        if BANGLA_NAME_LABEL in text:
            bangla_found = False
            # First try previous line
            for j in range(i - 1, -1, -1):
                prev = (lines[j].text or "").strip()
                if prev and has_bengali(prev) and not is_header_line(prev):
                    if ":" not in prev and not NAME_PREFIX.search(prev):
                        out["bangla_name"] = {"bbox": list(lines[j].bbox), "text": prev}
                        bangla_found = True
                    break
            # If not found before, try next line (e.g. "নাম:" then "মোঃ আটি ক রহমান")
            if not bangla_found and i + 1 < len(lines):
                next_line = (lines[i + 1].text or "").strip()
                if next_line and has_bengali(next_line) and not NAME_PREFIX.search(next_line):
                    out["bangla_name"] = {"bbox": list(lines[i + 1].bbox), "text": next_line}

    return out


# Load image
image = Image.open(IMAGE_PATH).convert("RGB")

# Load predictors
foundation = FoundationPredictor()
detector = DetectionPredictor()
recognizer = RecognitionPredictor(foundation)

# Run OCR
results = recognizer([image], det_predictor=detector)
lines = results[0].text_lines
text = "\n".join([ln.text for ln in lines])

# Save full OCR text
Path(OUTPUT_TXT).write_text(text, encoding="utf-8")
print(f"Result saved to: {OUTPUT_TXT}")

# Extract NID fields with bounding boxes
fields = extract_nid_fields(lines)

# JSON: bbox only (as requested) + text for reference
json_out = {
    k: {"bbox": v["bbox"], "text": v["text"]} if v else None
    for k, v in fields.items()
}

Path(OUTPUT_JSON).write_text(json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"NID fields (bbox + text) saved to: {OUTPUT_JSON}")
print(json.dumps(json_out, ensure_ascii=False, indent=2))