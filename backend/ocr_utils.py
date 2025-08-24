import cv2
import pytesseract
import re

# Force pytesseract to use the correct installed path (Windows fix)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Preprocess image before OCR
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Extract text using OCR
def extract_text(image_path):
    processed = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed)
    return text

# Extract structured fields (example: from RC or electricity bill)
def parse_fields(text):
    result = {}

    # Fuel type detection
    fuel_match = re.search(r"(Petrol|Diesel|Electric)", text, re.IGNORECASE)
    if fuel_match:
        result["fuel_type"] = fuel_match.group(1).capitalize()

    # Electricity units (e.g. "Units Consumed: 450 kWh")
    units_match = re.search(r"(\d+)\s*kWh", text)
    if units_match:
        result["monthly_units"] = int(units_match.group(1))

    return result

# Main function to process image and extract fields
def extract_ocr_data(image_path):
    text = extract_text(image_path)
    fields = parse_fields(text)
    return fields
