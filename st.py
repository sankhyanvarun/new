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
from io import BytesIO

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
if 'new_col_name' not in st.session_state:
    st.session_state.new_col_name = ""
if 'new_col_default' not in st.session_state:
    st.session_state.new_col_default = ""
if 'max_pages' not in st.session_state:
    st.session_state.max_pages = 70
if 'start_page' not in st.session_state:
    st.session_state.start_page = 1
if 'file_type' not in st.session_state:  # Track file type
    st.session_state.file_type = None

# Configure paths for cloud compatibility
poppler_path = '/usr/bin' if os.path.exists('/usr/bin') else None
tesseract_path = '/usr/bin/tesseract' if os.path.exists('/usr/bin/tesseract') else 'tesseract'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Hindi digit map
HINDI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]

# ========== COMMON FUNCTIONS ==========
def get_total_pages(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception:
        return 0

def enhance_image(img):
    """Apply advanced image enhancement for better OCR results"""
    try:
        # Convert to grayscale
        img = img.convert('L')
        
        # Convert to OpenCV format for advanced processing
        cv_img = np.array(img)
        
        # Apply denoising
        cv_img = cv2.fastNlMeansDenoising(cv_img, None, 10, 7, 21)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cv_img = clahe.apply(cv_img)
        
        # Convert back to PIL image
        img = Image.fromarray(cv_img)
    except Exception:
        # Fallback to basic enhancement if OpenCV fails
        img = img.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
    
    # Apply thresholding
    img = img.point(lambda p: 0 if p < 160 else 255)
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

def ocr_from_image(image: Image, language: str) -> str:
    """Perform OCR on an image"""
    try:
        enhanced_img = enhance_image(image)
        config = r'--oem 3 --psm 6'
        lang = "hin+eng" if language == "Hindi" else "eng"
        text = pytesseract.image_to_string(enhanced_img, config=config, lang=lang)
        if language == "Hindi":
            return normalize_hindi_digits(text)
        return text
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

# ========== TEXT EXTRACTION FUNCTIONS ==========
def extract_page_text(pdf_path: str, page_num: int, language: str) -> str:
    """Extract text from a single PDF page with enhanced OCR"""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            poppler_path=poppler_path,
            dpi=300,
            grayscale=True,
            thread_count=4
        )
        if images:
            img = images[0]
            return ocr_from_image(img, language) + "\n"
    except Exception as e:
        st.error(f"PDF Processing Error on page {page_num+1}: {str(e)}")
        return ""
    return ""

def extract_text_from_pages(pdf_path: str, page_indices: List[int], language: str) -> str:
    """Extract text from multiple PDF pages with OCR"""
    accumulated = ""
    for idx in page_indices:
        page_text = extract_page_text(pdf_path, idx, language)
        if page_text:
            accumulated += page_text
    return accumulated

# ========== HINDI-SPECIFIC FUNCTIONS ==========
def find_toc_page_indices_hindi(pdf_path: str, start_page: int = 0, max_search_pages: int = 20) -> List[int]:
    """Find pages with Hindi TOC keywords in PDF"""
    indices: List[int] = []
    try:
        total_pages = get_total_pages(pdf_path)
        search_limit = min(total_pages, start_page + max_search_pages)
        
        # Hindi TOC keywords
        toc_keywords = ["विषय सूची", "अनुक्रमणिका", "सूची", "विषय-सूची", "अनुक्रम", "सामग्री"]

        for i in range(start_page, search_limit):
            page_text = extract_page_text(pdf_path, i, "Hindi")
            # Check for any TOC keyword
            if any(keyword in page_text for keyword in toc_keywords):
                indices.append(i)
                
        return indices

    except Exception as e:
        st.error(f"Error finding Hindi TOC pages: {e}")
        return []

def extract_toc_hindi(text):
    """Extract TOC from Hindi text"""
    toc_start_pattern = r"(विषय[-\s]*सूची|अनुक्रमणिका|सूची|पृष्ठ|अध्याय|अंक|प्रकरण|सामग्री)"
    match = re.search(toc_start_pattern, text, re.IGNORECASE)
    if not match:
        return None

    toc_section = text[match.start():]
    toc_section = re.sub(r'[^\S\r\n]+', ' ', toc_section)  # Normalize whitespace
    lines = toc_section.split('\n')

    toc_entries = []
    current_title_lines = []
    digit_pattern = r'[\d०१२३४५६७८९]+$'  # Combined digit pattern
    multi_line_threshold = 4  # Minimum characters to consider as valid title

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        # Match page number at end of line
        page_match = re.search(digit_pattern, clean_line)
        page_number = page_match.group() if page_match else None

        # Handle different cases for TOC line formats
        if page_number:
            title_part = clean_line.rsplit(page_number, 1)[0].strip("–-—:. ")
            
            # Case 1: Page number with minimal title text + accumulated lines
            if current_title_lines and len(title_part) < multi_line_threshold:
                title = ' '.join(current_title_lines).strip()
                toc_entries.append({
                    "Title": title,
                    "Page": normalize_hindi_digits(page_number)
                })
                current_title_lines = []
            
            # Case 2: Page number with substantial title text
            else:
                if current_title_lines:
                    title = ' '.join(current_title_lines).strip()
                    toc_entries.append({
                        "Title": title,
                        "Page": normalize_hindi_digits(page_number)
                    })
                    current_title_lines = []
                if title_part and len(title_part) >= multi_line_threshold:
                    toc_entries.append({
                        "Title": title_part,
                        "Page": normalize_hindi_digits(page_number)
                    })
        else:
            # Accumulate lines without page numbers
            current_title_lines.append(clean_line)

    # Handle any remaining accumulated lines
    if current_title_lines:
        title = ' '.join(current_title_lines).strip()
        if len(title) >= multi_line_threshold:
            toc_entries.append({"Title": title, "Page": "?"})

    # Filter out invalid entries
    filtered_entries = [
        entry for entry in toc_entries
        if len(entry["Title"]) >= multi_line_threshold and not re.match(r'^\d+$', entry["Title"])
    ]
    
    return filtered_entries

# ========== ENGLISH-SPECIFIC FUNCTIONS ==========
def find_toc_page_indices_english(pdf_path: str, start_page: int = 0, max_search_pages: int = 20) -> List[int]:
    """Find pages with English TOC keywords in PDF"""
    indices: List[int] = []
    try:
        total_pages = get_total_pages(pdf_path)
        search_limit = min(total_pages, start_page + max_search_pages)

        for i in range(start_page, search_limit):
            page_text = extract_page_text(pdf_path, i, "English")
            lower_all = page_text.lower()
            
            # Look for English TOC keywords
            toc_keywords = ["contents", "table of contents", "toc", "index", "chapters"]
            if any(keyword in lower_all for keyword in toc_keywords):
                # Now see if parsing that page yields ≥ 2 valid TOC entries
                possible_entries = parse_toc_english(page_text)
                if len(possible_entries) >= 2:
                    indices.append(i)
                    # Don't break—TOC can span multiple consecutive pages

        return indices

    except Exception as e:
        st.error(f"Error finding TOC pages: {e}")
        return []

def parse_toc_english(text: str) -> List[Dict[str, str]]:
    """Parse TOC from English text (works for both PDF and image OCR)"""
    entries: List[Dict[str, str]] = []
    skip_terms = ["table of contents", "contents", "page", "toc", "chapter"]
    current_entry_lines = []  # Collect lines for the current TOC entry
    min_title_length = 4  # Minimum characters to consider as valid title

    # Improved pattern to match page numbers with various separators
    page_num_pattern = r'(\d+)[\s.]*$'

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        
        # Normalize Hindi digits in English documents too
        cleaned = normalize_hindi_digits(line)
        
        # Check if this line ends with a sequence of digits (page number)
        page_match = re.search(page_num_pattern, cleaned)
        if page_match:
            page_number = page_match.group(1)
            full_text = ""
            if current_entry_lines:
                # Combine buffered lines with current line
                full_text = " ".join(current_entry_lines) + " " + cleaned
                current_entry_lines = []  # Reset buffer
            else:
                full_text = cleaned
                
            # Skip entries that contain header terms
            lower_text = full_text.lower()
            if any(term in lower_text for term in skip_terms):
                continue
                
            # Extract title by removing the page number
            title_part = re.sub(page_num_pattern, '', full_text).strip(" .-–—:")
            
            # Only add if title is meaningful
            if len(title_part) >= min_title_length:
                entries.append({"Title": title_part, "Page": page_number})
        else:
            # Line doesn't end with page number → buffer it
            current_entry_lines.append(cleaned)
            
    return entries

# ========== IMAGE PROCESSING FUNCTIONS ==========
def process_image_file(uploaded_file, language: str):
    """Process uploaded image file and extract TOC"""
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform OCR
        extracted_text = ocr_from_image(image, language)
        st.session_state.raw_text = extracted_text
        
        # Extract TOC based on language
        if language == "Hindi":
            toc_entries = extract_toc_hindi(extracted_text) or []
        else:
            toc_entries = parse_toc_english(extracted_text) or []
        
        return toc_entries
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return []

# ========== UI AND MAIN APP ==========
def main():
    st.title("📖 PDF & Image TOC Extractor (Hindi + English)")
    
    # System verification
    st.subheader("System Verification")
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        st.success(f"Tesseract OCR {tesseract_version} is ready!")
    except:
        st.error("Tesseract not properly configured!")
    
    # Language selection
    st.session_state.language = st.radio(
        "Select Document Language:",
        ["Hindi", "English"],
        horizontal=True
    )
    
    # Configuration
    with st.expander("Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.extra_pages = st.slider(
                "Extra pages after TOC (PDF only)",
                min_value=0,
                max_value=20,
                value=st.session_state.extra_pages,
                help="TOC might span multiple pages"
            )
        with col2:
            st.session_state.max_pages = st.slider(
                "Max pages for large PDFs",
                min_value=5,
                max_value=200,
                value=st.session_state.max_pages,
                help="For large PDFs, only process first N pages"
            )
        
        # Starting page selection (PDF only)
        st.session_state.start_page = st.number_input(
            "Starting page for extraction (PDF only, 1-based)",
            min_value=1,
            max_value=1000,
            value=st.session_state.start_page,
            help="The page number from which to start searching for TOC"
        )
    
    # File upload section - now supports images too
    uploaded_file = st.file_uploader(
        "Upload PDF or Image file", 
        type=["pdf"] + SUPPORTED_IMAGE_FORMATS
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_image = file_extension in SUPPORTED_IMAGE_FORMATS
        
        # Display file info
        st.info(f"Uploaded file: {uploaded_file.name} ({'Image' if is_image else 'PDF'})")
        st.session_state.file_type = "image" if is_image else "pdf"
        
        if st.button("Extract Table of Contents"):
            with st.spinner("Processing file..."):
                try:
                    toc_entries = []
                    
                    if is_image:
                        # Process image file
                        toc_entries = process_image_file(uploaded_file, st.session_state.language)
                    else:
                        # Process PDF file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            pdf_path = os.path.join(temp_dir, "uploaded.pdf")
                            with open(pdf_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Check if PDF is large
                            file_size = os.path.getsize(pdf_path)
                            is_large_pdf = file_size > 4 * 1024 * 1024  # 4 MB
                            truncated_path = None
                            
                            if is_large_pdf:
                                st.info(f"Large PDF detected ({file_size/(1024*1024):.2f} MB). Processing first {st.session_state.max_pages} pages.")
                                truncated_path = os.path.join(temp_dir, "truncated.pdf")
                                truncate_pdf(pdf_path, truncated_path, max_pages=st.session_state.max_pages)
                                extraction_path = truncated_path
                            else:
                                extraction_path = pdf_path
                            
                            # Language-specific extraction
                            start_page_index = st.session_state.start_page - 1  # Convert to 0-based index
                            
                            if st.session_state.language == "Hindi":
                                # Find TOC pages
                                toc_indices = find_toc_page_indices_hindi(
                                    extraction_path, 
                                    start_page=start_page_index
                                )
                                
                                # Expand TOC indices
                                expanded_indices = set()
                                for i in toc_indices:
                                    for offset in range(0, st.session_state.extra_pages + 1):
                                        expanded_indices.add(i + offset)
                                
                                # If no TOC found, use starting page range
                                if not expanded_indices:
                                    total_pages = get_total_pages(extraction_path)
                                    end_page = min(start_page_index + 20, total_pages)
                                    expanded_indices = set(range(start_page_index, end_page))
                                
                                # Extract text from identified pages
                                extracted_text = extract_text_from_pages(
                                    extraction_path, 
                                    sorted(expanded_indices),
                                    "Hindi"
                                )
                                st.session_state.raw_text = extracted_text
                                
                                # Attempt TOC extraction
                                toc_entries = extract_toc_hindi(extracted_text) or []
                            else:  # English
                                # Find TOC pages
                                toc_indices = find_toc_page_indices_english(
                                    extraction_path, 
                                    start_page=start_page_index
                                )
                                
                                # Expand TOC indices
                                expanded_indices = set()
                                for i in toc_indices:
                                    for offset in range(0, st.session_state.extra_pages + 1):
                                        expanded_indices.add(i + offset)
                                
                                # If no TOC found, use starting page range
                                if not expanded_indices:
                                    total_pages = get_total_pages(extraction_path)
                                    end_page = min(start_page_index + 20, total_pages)
                                    expanded_indices = set(range(start_page_index, end_page))
                                
                                # Extract text from identified pages
                                extracted_text = extract_text_from_pages(
                                    extraction_path, 
                                    sorted(expanded_indices),
                                    "English"
                                )
                                st.session_state.raw_text = extracted_text
                                
                                # Attempt TOC extraction
                                toc_entries = parse_toc_english(extracted_text) or []
                    
                    # Process results
                    if toc_entries:
                        st.success(f"Successfully extracted {len(toc_entries)} TOC entries!")
                        st.session_state.toc_df = pd.DataFrame(toc_entries)
                    else:
                        st.warning("No TOC found in the document")
                        st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
                        
                    # Show raw text extraction message
                    if st.session_state.file_type == "pdf":
                        st.info(f"Extracted {len(st.session_state.raw_text)} characters from PDF")
                    else:
                        st.info(f"Extracted {len(st.session_state.raw_text)} characters from image")
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
    
    # Edit TOC Section
    if not st.session_state.toc_df.empty:
        st.subheader("Table of Contents")
        
        # Display non-editable preview
        st.dataframe(st.session_state.toc_df, use_container_width=True)
        
        # Edit mode toggle
        if not st.session_state.edit_mode:
            if st.button("✏️ Edit TOC"):
                st.session_state.edit_mode = True
                st.session_state.backup_df = st.session_state.toc_df.copy()
        else:
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
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Changes"):
                    st.session_state.toc_df = edited_df
                    st.session_state.edit_mode = False
                    st.success("Changes saved!")
            with col2:
                if st.button("❌ Discard Changes"):
                    st.session_state.toc_df = st.session_state.backup_df
                    st.session_state.edit_mode = False
                    st.info("Changes discarded")
            
            # Advanced editing options
            with st.expander("🔄 Advanced Editing Tools", expanded=True):
                # Row insertion at specific position
                st.subheader("Insert Row at Specific Position")
                with st.form("row_insert_form"):
                    insert_position = st.number_input(
                        "Row Position (1-based)",
                        min_value=1,
                        max_value=len(edited_df)+1,
                        value=len(edited_df)+1,
                        help="Position where the new row will be inserted"
                    )
                    new_title = st.text_input("Title")
                    new_page = st.text_input("Page")
                    
                    if st.form_submit_button("Insert Row"):
                        # Create new row as DataFrame
                        new_row = pd.DataFrame([[new_title, new_page]], columns=edited_df.columns)
                        
                        # Convert 1-based position to 0-based index
                        pos_index = insert_position - 1
                        
                        # Insert row
                        top_part = edited_df.iloc[:pos_index]
                        bottom_part = edited_df.iloc[pos_index:]
                        edited_df = pd.concat([top_part, new_row, bottom_part]).reset_index(drop=True)
                        st.session_state.toc_df = edited_df
                        st.success(f"Inserted new row at position {insert_position}")
                
                # Column insertion
                st.subheader("Add New Column")
                with st.form("column_insert_form"):
                    col_name = st.text_input("Column Name", value=st.session_state.new_col_name)
                    default_value = st.text_input("Default Value", value=st.session_state.new_col_default)
                    
                    if st.form_submit_button("Add Column"):
                        if col_name in edited_df.columns:
                            st.error(f"Column '{col_name}' already exists!")
                        else:
                            # Add new column with default value
                            edited_df[col_name] = default_value
                            st.session_state.toc_df = edited_df
                            st.session_state.new_col_name = col_name
                            st.session_state.new_col_default = default_value
                            st.success(f"Added new column: {col_name}")
    
    # Handle case where DataFrame is empty
    elif not st.session_state.toc_df.empty and st.session_state.toc_df.columns.tolist() != ["Title", "Page"]:
        st.warning("TOC data is in an invalid format. Resetting...")
        st.session_state.toc_df = pd.DataFrame(columns=["Title", "Page"])
    
    # Raw text section
    if 'raw_text' in st.session_state and st.session_state.raw_text:
        with st.expander("View Raw Extracted Text"):
            st.text_area("Raw OCR Output", st.session_state.raw_text, height=300)
            st.info(f"Total characters: {len(st.session_state.raw_text)}")
    
    # Download section
    if not st.session_state.toc_df.empty:
        st.subheader("Download Final TOC")
        csv = st.session_state.toc_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download TOC as CSV",
            data=csv,
            file_name="table_of_contents.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
