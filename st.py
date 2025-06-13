import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import tempfile
import numpy as np
import cv2

# Configure paths
poppler_path = '/usr/bin' if os.path.exists('/usr/bin') else None
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' if os.path.exists('/usr/bin/tesseract') else 'tesseract'

# Enhanced image preprocessing
def enhance_image(img):
    """
    Apply advanced image preprocessing for better OCR results
    Includes: denoising, contrast enhancement, sharpening, and binarization
    """
    # Convert to grayscale
    img = img.convert('L')
    
    # Convert to OpenCV format for advanced processing
    cv_img = np.array(img)
    
    # Apply denoising
    cv_img = cv2.fastNlMeansDenoising(cv_img, None, 10, 7, 21)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cv_img = clahe.apply(cv_img)
    
    # Convert back to PIL image
    img = Image.fromarray(cv_img)
    
    # Apply sharpening
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    # Apply adaptive thresholding
    img = img.point(lambda p: 0 if p < 180 else 255)
    
    # Remove salt & pepper noise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img

def extract_text_from_pdf(pdf_path, language='eng', max_pages=70):
    """
    Extract text from PDF with enhanced OCR processing
    - Processes only first 'max_pages' for large PDFs
    - Returns text and list of processed images
    """
    try:
        # Get total pages
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)
        
        # Determine pages to process
        pages_to_process = min(total_pages, max_pages)
        st.info(f"Processing first {pages_to_process} of {total_pages} pages")
        
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            first_page=1,
            last_page=pages_to_process,
            poppler_path=poppler_path,
            dpi=300,
            grayscale=True,
            thread_count=4
        )
        
        if not images:
            st.error("No pages converted")
            return "", []
        
        processed_images = []
        full_text = ""
        
        for i, img in enumerate(images):
            # Apply advanced preprocessing
            enhanced_img = enhance_image(img)
            processed_images.append(enhanced_img)
            
            # OCR with language selection
            lang = 'hin+eng' if language == 'Hindi' else 'eng'
            page_text = pytesseract.image_to_string(
                enhanced_img,
                lang=lang,
                config='--psm 6 --oem 3'
            )
            
            full_text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
            
            # Early stop if we've found TOC in first pages
            if i < 10 and ("contents" in page_text.lower() or "सूची" in page_text):
                st.success(f"Found TOC indicators on page {i+1}")
                break
        
        return full_text, processed_images
        
    except Exception as e:
        st.error(f"Extraction failed: {str(e)}")
        return "", []

# Streamlit UI
st.title("📖 Advanced PDF TOC Extractor")

# System check
st.subheader("System Verification")
try:
    st.write(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    st.write(f"Poppler path: {poppler_path or 'System default'}")
    st.success("OCR system is ready!")
except:
    st.error("System verification failed - check dependencies")

# Settings
with st.sidebar.expander("Processing Settings"):
    language = st.radio("Document Language", ["English", "Hindi"])
    max_pages = st.slider("Max pages to process", 10, 200, 70, 
                          help="For large PDFs, only process first N pages")
    show_images = st.checkbox("Show processed images", True)
    show_text = st.checkbox("Show extracted text", True)

# File upload
uploaded_file = st.file_uploader("Upload PDF document", type="pdf")

if uploaded_file:
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process PDF
        text, processed_images = extract_text_from_pdf(
            tmp_path, 
            language=language,
            max_pages=max_pages
        )
        
        # Display results
        if text:
            st.success(f"Extracted {len(text)} characters from {len(processed_images)} pages")
            
            # TOC detection
            toc_found = False
            toc_keywords = {
                "English": ["contents", "table of contents", "toc", "index"],
                "Hindi": ["सूची", "विषय", "अनुक्रमणिका", "अनुक्रमणिका"]
            }
            
            for keyword in toc_keywords[language]:
                if keyword.lower() in text.lower():
                    st.success(f"Found TOC indicator: '{keyword}'")
                    toc_found = True
                    break
            
            if not toc_found:
                st.warning("No clear TOC indicators found. Try increasing page limit or check document.")
            
            # Show processed images
            if show_images and processed_images:
                st.subheader("Processed Page Images")
                cols = st.columns(3)
                for i, img in enumerate(processed_images):
                    if i < 9:  # Show max 9 images
                        with cols[i % 3]:
                            st.image(img, caption=f"Page {i+1}", use_column_width=True)
            
            # Show extracted text
            if show_text:
                with st.expander("Extracted Text", expanded=True):
                    st.text(text)
            
            # Download button
            st.download_button(
                label="Download Extracted Text",
                data=text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
        else:
            st.error("No text extracted - document may be image-based or low quality")
            
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
