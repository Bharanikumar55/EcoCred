import pytesseract
import cv2
from pdf2image import convert_from_path
import os
import re

# Change this to your Poppler path if not in PATH
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"

def extract_text_from_file(file_path):
    """Convert PDF to images (if needed), then OCR extract text."""
    text = ""
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages):
            img_path = f"temp_page_{i}.png"
            page.save(img_path, "PNG")
            img = cv2.imread(img_path)
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
            os.remove(img_path)
    else:
        img = cv2.imread(file_path)
        text = pytesseract.image_to_string(img)
    return text

def extract_electricity_bill(file_path):
    """Extract electricity units from text using regex."""
    text = extract_text_from_file(file_path)
    m = re.search(r"(\d+)\s*(kwh|units?)", text.lower())
    if m:
        return int(m.group(1))
    numbers = [int(s) for s in re.findall(r"\d+", text)]
    return max(numbers) if numbers else 0

def extract_fuel_type(file_path):
    """Detect fuel type from RC."""
    text = extract_text_from_file(file_path).lower()
    if "electric" in text:
        return "Electric"
    if "diesel" in text:
        return "Diesel"
    if "petrol" in text:
        return "Petrol"
    return "Unknown"

def calculate_eco_score(fuel_type):
    """Simple eco score mapping."""
    f = str(fuel_type).lower()
    if f == "electric":
        return 1.0
    if f == "petrol":
        return 0.5
    if f == "diesel":
        return 0.3
    return 0.4
