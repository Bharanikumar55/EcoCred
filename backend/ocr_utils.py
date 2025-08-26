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
        print(f"[INFO] Converting PDF -> Images: {file_path}")
        pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages):
            img_path = f"temp_page_{i}.png"
            page.save(img_path, "PNG")
            page_text = pytesseract.image_to_string(cv2.imread(img_path))
            print(f"[DEBUG] OCR Output Page {i}:\n{page_text}\n{'-'*40}")
            text += page_text + "\n"
            os.remove(img_path)
    else:
        img = cv2.imread(file_path)
        text = pytesseract.image_to_string(img)
        print(f"[DEBUG] OCR Output Image:\n{text}\n{'-'*40}")
    return text

def extract_electricity_bill(file_path):
    """Extract electricity units from text using regex"""
    text = extract_text_from_file(file_path)
    # look for numbers followed by 'kwh' or 'units'
    match = re.search(r"(\d+)\s*(kwh|units?)", text.lower())
    if match:
        return int(match.group(1))
    # fallback: pick the largest integer from text
    numbers = [int(s) for s in re.findall(r"\d+", text)]
    return max(numbers) if numbers else 0

def extract_fuel_type(file_path):
    """Detect fuel type from RC"""
    text = extract_text_from_file(file_path).lower()
    if "electric" in text:
        return "Electric"
    elif "diesel" in text:
        return "Diesel"
    elif "petrol" in text:
        return "Petrol"
    return "Unknown"

def calculate_eco_score(fuel_type):
    """Simple eco score mapping"""
    if fuel_type.lower() == "electric":
        return 1.0
    elif fuel_type.lower() == "petrol":
        return 0.5
    elif fuel_type.lower() == "diesel":
        return 0.3
    else:
        return 0.4
