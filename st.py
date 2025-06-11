import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import re
import os
import pandas as pd
import tempfile
import PyPDF2
import numpy as np
from typing import List, Dict
import subprocess
import sys

def install_system_dependencies():
    try:
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 
                       'tesseract-ocr',
                       'tesseract-ocr-hin',
                       'tesseract-ocr-eng',
                       'poppler-utils',
                       'libsm6',
                       'libxext6',
                       'ffmpeg'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)

# Run at startup
install_system_dependencies()
# Set page config
st.set_page_config(page_title="Enhanced Multi-Language TOC Extractor", layout="wide")

# Initialize session state
if 'toc_df' not in st.session_state:
    st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'language' not in st.session_state:
    st.session_state.language = "Hindi"
if 'extra_pages' not in st.session_state:
    st.session_state.extra_pages = 2
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False

# Set paths based on environment
if os.path.exists('/usr/bin/tesseract'):  # Streamlit Cloud
    st.session_state.poppler_path = '/usr/bin'
    st.session_state.tesseract_path = '/usr/bin/tesseract'
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
else:  # Local development
    st.session_state.poppler_path = 'poppler/bin' if os.path.exists('poppler/bin') else None
    st.session_state.tesseract_path = 'Tesseract-OCR/tesseract.exe' if os.path.exists('Tesseract-OCR') else None
    if st.session_state.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = st.session_state.tesseract_path

# Hindi digit map
HINDI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# ========== COMMON FUNCTIONS ==========
def check_tesseract():
    try:
        langs = pytesseract.get_languages(config='')
        st.info(f"Available Tesseract languages: {langs}")
        return 'hin' in langs and 'eng' in langs
    except Exception as e:
        st.error(f"Tesseract check failed: {str(e)}")
        return False

def get_total_pages(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return 0

def enhance_image(img):
    """Apply image enhancement for better OCR results"""
    img = img.convert('L')  # Convert to grayscale
    img = img.filter(ImageFilter.MedianFilter())  # Reduce noise
    enhancer = ImageEnhance.Contrast(img)  # Enhance contrast
    img = enhancer.enhance(2)  # Increase contrast
    img = img.point(lambda p: 0 if p < 150 else 255)  # Thresholding
    return img

def truncate_pdf(input_path: str, output_path: str, max_pages: int = 70) -> None:
    reader = PyPDF2.PdfReader(input_path)
    writer = PyPDF2.PdfWriter()
    
    num_pages = len(reader.pages)
    pages_to_keep = min(max_pages, num_pages)
    
    for i in range(pages_to_keep):
        writer.add_page(reader.pages[i])
    
    with open(output_path, "wb") as f:
        writer.write(f)

def normalize_hindi_digits(text):
    return ''.join(HINDI_DIGIT_MAP.get(ch, ch) for ch in text)

# ========== TEXT EXTRACTION FUNCTIONS ==========
def extract_page_text(pdf_path: str, page_num: int, language: str) -> str:
    """Extract text from a single page with OCR and enhancement"""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=st.session_state.poppler_path,
            dpi=300 if language == "Hindi" else 400,
            grayscale=True,
        )
        if images:
            img = images[0]
            enhanced_img = enhance_image(img)
            config = r'--oem 1 --psm 6'
            lang = "hin+eng" if language == "Hindi" else "eng"
            page_text = pytesseract.image_to_string(enhanced_img, config=config, lang=lang)
            
            if language == "Hindi":
                return normalize_hindi_digits(page_text) + "\n"
            return page_text + "\n"
    except Exception as e:
        st.error(f"OCR Error on page {page_num}: {str(e)}")
        return ""
    return ""

[Rest of your existing functions remain exactly the same...]

def main():
    st.title("📖 Enhanced Multi-Language PDF TOC Extractor")
    
    # Environment check
    if not check_tesseract():
        st.error("Tesseract OCR not properly configured! Check deployment logs.")
        st.stop()

    [Rest of your existing main() function remains the same...]

if __name__ == "__main__":
    main()
