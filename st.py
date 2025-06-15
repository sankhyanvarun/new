import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import re
import os
import pandas as pd
import tempfile
import PyPDF2
import numpy as np
import cv2
from typing import List, Dict
import time

# Set page config
st.set_page_config(page_title="Professional PDF TOC Extractor", layout="wide")

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
if 'new_col_name' not in st.session_state:
    st.session_state.new_col_name = ""
if 'new_col_default' not in st.session_state:
    st.session_state.new_col_default = ""
if 'max_pages' not in st.session_state:
    st.session_state.max_pages = 70
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Configure paths for cloud compatibility
poppler_path = '/usr/bin' if os.path.exists('/usr/bin') else None
tesseract_path = '/usr/bin/tesseract' if os.path.exists('/usr/bin/tesseract') else 'tesseract'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Hindi digit map
HINDI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# ========== ENHANCED IMAGE PROCESSING ==========
def enhance_image(img):
    """Professional-grade image enhancement for optimal OCR results"""
    try:
        # Convert to OpenCV format (BGR to RGB)
        cv_img = np.array(img.convert('RGB'))
        cv_img = cv_img[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Apply advanced denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21, 
            10
        )
        
        # Apply morphological operations to clean up text
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        processed = clahe.apply(processed)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(processed, -1, kernel)
        
        # Convert back to PIL image
        result = Image.fromarray(sharpened)
        return result
        
    except Exception as e:
        # Fallback to basic enhancement if OpenCV fails
        img = img.convert('L')
        img = img.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(3.0)
        img = img.point(lambda p: 0 if p < 180 else 255)
        return img

# ========== OPTIMIZED TEXT EXTRACTION ==========
def extract_page_text(pdf_path: str, page_num: int, language: str, progress_bar=None) -> str:
    """Advanced text extraction with progress tracking"""
    try:
        start_time = time.time()
        
        # Convert PDF to image
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=poppler_path,
            dpi=350,  # Higher DPI for better quality
            grayscale=True,
            thread_count=4
        )
        
        if not images:
            return ""
        
        img = images[0]
        
        # Apply professional enhancement
        enhanced_img = enhance_image(img)
        
        # OCR configuration
        config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        lang = "hin+eng" if language == "Hindi" else "eng"
        
        # Perform OCR
        page_text = pytesseract.image_to_string(
            enhanced_img, 
            config=config, 
            lang=lang
        )
        
        # Normalize Hindi digits
        if language == "Hindi":
            page_text = normalize_hindi_digits(page_text)
        
        # Update progress
        if progress_bar:
            st.session_state.progress += 1
            progress_bar.progress(st.session_state.progress)
        
        proc_time = time.time() - start_time
        st.toast(f"Processed page {page_num+1} in {proc_time:.1f}s", icon="⏱️")
        
        return page_text + "\n"
        
    except Exception as e:
        st.error(f"Error processing page {page_num+1}: {str(e)}")
        return ""

# ========== COMMON FUNCTIONS ==========
def get_total_pages(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return 0

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

def extract_text_from_pages(pdf_path: str, page_indices: List[int], language: str) -> str:
    """Extract text from multiple pages with progress tracking"""
    accumulated = ""
    total_pages = len(page_indices)
    st.session_state.progress = 0
    progress_bar = st.progress(0)
    
    for idx in page_indices:
        page_text = extract_page_text(pdf_path, idx, language, progress_bar)
        if page_text:
            accumulated += page_text
    
    return accumulated

# ========== TOC EXTRACTION FUNCTIONS ==========
# [Keep your existing Hindi and English TOC extraction functions unchanged]

# ========== UI AND MAIN APP ==========
def main():
    st.title("📖 Professional PDF TOC Extractor")
    st.markdown("""
    **Industry-grade PDF processing with:**
    - Advanced image enhancement for optimal OCR
    - Multi-threaded text extraction
    - Intelligent TOC detection
    - Progress tracking and real-time feedback
    """)
    
    # System verification
    with st.expander("System Configuration", expanded=True):
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            st.success(f"Tesseract OCR {tesseract_version}")
            st.info(f"Poppler path: {poppler_path or 'System default'}")
            
            # Display available languages
            langs = pytesseract.get_languages()
            st.write(f"Available languages: {', '.join(langs)}")
            
        except Exception as e:
            st.error(f"System error: {str(e)}")
    
    # Language selection
    st.session_state.language = st.radio(
        "Select PDF Language:",
        ["Hindi", "English"],
        horizontal=True
    )
    
    # Processing settings
    with st.expander("Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.extra_pages = st.slider(
                "Extra pages after TOC",
                min_value=0,
                max_value=20,
                value=st.session_state.extra_pages,
                help="TOC might span multiple pages"
            )
        with col2:
            st.session_state.max_pages = st.slider(
                "Max pages for large PDFs",
                min_value=10,
                max_value=200,
                value=st.session_state.max_pages,
                help="For large PDFs, only process first N pages"
            )
    
    # File upload section
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf", accept_multiple_files=False)
    
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "uploaded.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Extract Table of Contents", type="primary"):
                # Create processing container
                processing_container = st.container()
                
                with processing_container:
                    st.subheader("Processing Status")
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    try:
                        # Check if PDF is large
                        file_size = os.path.getsize(pdf_path)
                        is_large_pdf = file_size > 50 * 1024 * 1024  # 50 MB
                        truncated_path = None
                        
                        if is_large_pdf:
                            status_text.info(f"Large PDF detected ({file_size/(1024*1024):.2f} MB). Processing first {st.session_state.max_pages} pages.")
                            truncated_path = os.path.join(temp_dir, "truncated.pdf")
                            truncate_pdf(pdf_path, truncated_path, max_pages=st.session_state.max_pages)
                            extraction_path = truncated_path
                        else:
                            extraction_path = pdf_path
                        
                        # Get total pages
                        total_pages = get_total_pages(extraction_path)
                        status_text.info(f"Processing PDF with {total_pages} pages...")
                        
                        # Language-specific extraction
                        if st.session_state.language == "Hindi":
                            # Find TOC pages
                            status_text.info("Searching for Hindi TOC pages...")
                            toc_indices = find_toc_page_indices_hindi(extraction_path)
                            
                            # Expand TOC indices
                            expanded_indices = set()
                            for i in toc_indices:
                                for offset in range(0, st.session_state.extra_pages + 1):
                                    expanded_indices.add(i + offset)
                            
                            # If no TOC found, use first 20 pages
                            if not expanded_indices:
                                expanded_indices = set(range(0, min(20, total_pages)))
                                status_text.warning("No TOC detected. Processing first 20 pages.")
                            
                            # Extract text from identified pages
                            status_text.info(f"Extracting text from {len(expanded_indices)} pages...")
                            extracted_text = extract_text_from_pages(
                                extraction_path, 
                                sorted(expanded_indices),
                                "Hindi"
                            )
                            st.session_state.raw_text = extracted_text
                            
                            # Attempt TOC extraction
                            status_text.info("Parsing TOC entries...")
                            toc_entries = extract_toc_hindi(extracted_text) or []
                        else:  # English
                            # Find TOC pages
                            status_text.info("Searching for English TOC pages...")
                            toc_indices = find_toc_page_indices_english(extraction_path)
                            
                            # Expand TOC indices
                            expanded_indices = set()
                            for i in toc_indices:
                                for offset in range(0, st.session_state.extra_pages + 1):
                                    expanded_indices.add(i + offset)
                            
                            # If no TOC found, use first 20 pages
                            if not expanded_indices:
                                expanded_indices = set(range(0, min(20, total_pages)))
                                status_text.warning("No TOC detected. Processing first 20 pages.")
                            
                            # Extract text from identified pages
                            status_text.info(f"Extracting text from {len(expanded_indices)} pages...")
                            extracted_text = extract_text_from_pages(
                                extraction_path, 
                                sorted(expanded_indices),
                                "English"
                            )
                            st.session_state.raw_text = extracted_text
                            
                            # Attempt TOC extraction
                            status_text.info("Parsing TOC entries...")
                            toc_entries = parse_toc_english(extracted_text) or []
                        
                        # Process results
                        if toc_entries:
                            status_text.success(f"Successfully extracted {len(toc_entries)} TOC entries!")
                            st.session_state.toc_df = pd.DataFrame(toc_entries)
                        else:
                            status_text.warning("No TOC found in the document")
                            st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
                            
                        # Show raw text extraction message
                        status_text.info(f"Extracted {len(st.session_state.raw_text)} characters from {len(expanded_indices)} pages")
                            
                    except Exception as e:
                        status_text.error(f"Processing failed: {str(e)}")
    
    # Display results
    if not st.session_state.toc_df.empty:
        st.subheader("Extracted Table of Contents")
        st.dataframe(st.session_state.toc_df, use_container_width=True, height=400)
        
        # Edit controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ Edit TOC", use_container_width=True):
                st.session_state.edit_mode = True
                st.session_state.backup_df = st.session_state.toc_df.copy()
        with col2:
            csv = st.session_state.toc_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download TOC as CSV",
                data=csv,
                file_name="table_of_contents.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Edit mode
    if st.session_state.edit_mode:
        st.subheader("Edit Mode")
        
        # Display editable dataframe
        edited_df = st.data_editor(
            st.session_state.toc_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Chapter Title", width="large"),
                "Page": st.column_config.TextColumn("Page Number", width="small")
            }
        )
        
        # Edit controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                st.session_state.toc_df = edited_df
                st.session_state.edit_mode = False
                st.success("Changes saved!")
        with col2:
            if st.button("❌ Discard Changes", use_container_width=True):
                st.session_state.toc_df = st.session_state.backup_df
                st.session_state.edit_mode = False
                st.info("Changes discarded")
    
    # Raw text section
    if 'raw_text' in st.session_state and st.session_state.raw_text:
        with st.expander("View Raw Extracted Text"):
            st.text_area("Raw OCR Output", st.session_state.raw_text, height=300)
            st.info(f"Total characters: {len(st.session_state.raw_text)}")

if __name__ == "__main__":
    main()
