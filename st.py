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
    st.session_state.max_pages = 70  # Default max pages for large PDFs

# Configure paths for cloud compatibility
poppler_path = '/usr/bin' if os.path.exists('/usr/bin') else None
tesseract_path = '/usr/bin/tesseract' if os.path.exists('/usr/bin/tesseract') else 'tesseract'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Hindi digit map
HINDI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

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

# ========== TEXT EXTRACTION FUNCTIONS ==========
def extract_page_text(pdf_path: str, page_num: int, language: str) -> str:
    """Extract text from a single page with enhanced OCR"""
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
            enhanced_img = enhance_image(img)
            
            # OCR configuration
            config = r'--oem 3 --psm 6'  # Improved OCR engine mode
            lang = "hin+eng" if language == "Hindi" else "eng"  # Combined languages for better accuracy
            page_text = pytesseract.image_to_string(enhanced_img, config=config, lang=lang)
            
            # Normalize Hindi digits
            if language == "Hindi":
                return normalize_hindi_digits(page_text) + "\n"
            return page_text + "\n"
    except Exception as e:
        st.error(f"OCR Error on page {page_num+1}: {str(e)}")
        return ""
    return ""

def extract_text_from_pages(pdf_path: str, page_indices: List[int], language: str) -> str:
    """Extract text from multiple pages with OCR"""
    accumulated = ""
    for idx in page_indices:
        page_text = extract_page_text(pdf_path, idx, language)
        if page_text:
            accumulated += page_text
    return accumulated

# ========== HINDI-SPECIFIC FUNCTIONS ==========
def find_toc_page_indices_hindi(pdf_path: str, max_search_pages: int = 20) -> List[int]:
    """Find pages with Hindi TOC keywords"""
    indices: List[int] = []
    try:
        total_pages = get_total_pages(pdf_path)
        search_limit = min(total_pages, max_search_pages)
        
        # Hindi TOC keywords
        toc_keywords = ["विषय सूची", "अनुक्रमणिका", "सूची", "विषय-सूची", "अनुक्रम", "सामग्री"]

        for i in range(search_limit):
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
            if current_title_lines and len(title_part) < 5:
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
                if title_part:
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
        if len(title) > 5:
            toc_entries.append({"Title": title, "Page": "?"})

    # Filter out invalid entries
    filtered_entries = [
        entry for entry in toc_entries
        if len(entry["Title"]) >= 5 and not re.match(r'^\d+$', entry["Title"])
    ]
    
    return filtered_entries

# ========== ENGLISH-SPECIFIC FUNCTIONS ==========
def find_toc_page_indices_english(pdf_path: str, max_search_pages: int = 20) -> List[int]:
    indices: List[int] = []
    try:
        total_pages = get_total_pages(pdf_path)
        search_limit = min(total_pages, max_search_pages)

        for i in range(search_limit):
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
    entries: List[Dict[str, str]] = []
    skip_terms = ["table of contents", "contents", "page", "toc", "chapter"]
    current_entry_lines = []  # Collect lines for the current TOC entry

    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        
        # Normalize Hindi digits in English documents too
        cleaned = normalize_hindi_digits(line)
        
        # Check if this line ends with a sequence of digits (page number)
        if re.search(r'\d+\s*$', cleaned):
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
                
            # Attempt to split into chapter and page number
            m = re.match(r'^(.*?)[\s.\-]+\s*(\d+)\s*$', full_text)
            if not m:
                # Fallback: match any trailing digits
                m = re.match(r'^(.*?)(\d+)\s*$', full_text)
                
            if m:
                chapter = m.group(1).strip()
                page_no = m.group(2).strip()
                entries.append({"Title": chapter, "Page": page_no})
        else:
            # Line doesn't end with page number → buffer it
            current_entry_lines.append(cleaned)
            
    return entries

# ========== UI AND MAIN APP ==========
def main():
    st.title("📖 Enhanced Multi-Language PDF TOC Extractor")
    st.markdown("""
    Extract Table of Contents from Hindi or English PDFs:
    - **Advanced OCR**: Professional-grade text extraction with image enhancement
    - **Cloud Optimized**: Works seamlessly on Streamlit Cloud
    - **Large PDF Support**: Automatically processes first 70 pages for large documents
    - **Multi-Language**: Specialized handling for both Hindi and English documents
    """)
    
    # System verification
    st.subheader("System Verification")
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        st.success(f"Tesseract OCR {tesseract_version} is ready!")
        st.write(f"Poppler path: {poppler_path or 'System default'}")
    except:
        st.error("Tesseract not properly configured!")
    
    # Language selection
    st.session_state.language = st.radio(
        "Select PDF Language:",
        ["Hindi", "English"],
        horizontal=True
    )
    
    # Configuration
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
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "uploaded.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Extract Table of Contents"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Check if PDF is large
                        file_size = os.path.getsize(pdf_path)
                        is_large_pdf = file_size > 50 * 1024 * 1024  # 50 MB
                        truncated_path = None
                        
                        if is_large_pdf:
                            st.info(f"Large PDF detected ({file_size/(1024*1024):.2f} MB). Processing first {st.session_state.max_pages} pages.")
                            truncated_path = os.path.join(temp_dir, "truncated.pdf")
                            truncate_pdf(pdf_path, truncated_path, max_pages=st.session_state.max_pages)
                            extraction_path = truncated_path
                        else:
                            extraction_path = pdf_path
                        
                        # Language-specific extraction
                        if st.session_state.language == "Hindi":
                            # Find TOC pages
                            toc_indices = find_toc_page_indices_hindi(extraction_path)
                            
                            # Expand TOC indices
                            expanded_indices = set()
                            for i in toc_indices:
                                for offset in range(0, st.session_state.extra_pages + 1):
                                    expanded_indices.add(i + offset)
                            
                            # If no TOC found, use first 20 pages
                            if not expanded_indices:
                                total_pages = get_total_pages(extraction_path)
                                expanded_indices = set(range(0, min(20, total_pages)))
                            
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
                            toc_indices = find_toc_page_indices_english(extraction_path)
                            
                            # Expand TOC indices
                            expanded_indices = set()
                            for i in toc_indices:
                                for offset in range(0, st.session_state.extra_pages + 1):
                                    expanded_indices.add(i + offset)
                            
                            # If no TOC found, use first 20 pages
                            if not expanded_indices:
                                total_pages = get_total_pages(extraction_path)
                                expanded_indices = set(range(0, min(20, total_pages)))
                            
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
                        st.info(f"Extracted {len(st.session_state.raw_text)} characters from {len(expanded_indices)} pages")
                            
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
    
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
                st.experimental_rerun()
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
            col1, col2, col3 = st.columns(3)
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
            with st.expander("🔄 Advanced Editing Tools"):
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
                        new_row = pd.DataFrame([[new_title, new_page]], columns=["Title", "Page"])
                        
                        # Convert 1-based position to 0-based index
                        pos_index = insert_position - 1
                        
                        # Insert row
                        top_part = edited_df.iloc[:pos_index]
                        bottom_part = edited_df.iloc[pos_index:]
                        edited_df = pd.concat([top_part, new_row, bottom_part]).reset_index(drop=True)
                        st.session_state.toc_df = edited_df
                        st.success(f"Inserted new row at position {insert_position}")
                        st.experimental_rerun()
                
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
                            st.experimental_rerun()
    
    # Raw text section for both languages
    if 'raw_text' in st.session_state and st.session_state.raw_text:
        with st.expander("View Raw Extracted Text"):
            st.text_area("Raw OCR Output", st.session_state.raw_text, height=300)
            st.info(f"Total characters: {len(st.session_state.raw_text)}")
    
    # Download section (always visible if we have data)
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
