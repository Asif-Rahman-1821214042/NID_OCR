import sys
from pathlib import Path

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

IMAGE_PATH = "sajid.jpeg"
OUTPUT_TXT = "ocr_result.txt"  # OCR text saved here

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")

# Load predictors
foundation = FoundationPredictor()
detector = DetectionPredictor()
recognizer = RecognitionPredictor(foundation)

# Run OCR (Surya auto-handles multilingual text, including bn + en)
results = recognizer([image], det_predictor=detector)

# Get plain text (line-by-line)
lines = results[0].text_lines
text = "\n".join([ln.text for ln in lines])

# Save result to txt file (UTF-8)
Path(OUTPUT_TXT).write_text(text, encoding="utf-8")
print(f"Result saved to: {OUTPUT_TXT}")

print("OCR text (terminal):")
print(text)