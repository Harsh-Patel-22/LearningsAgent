import fitz  # PyMuPDF
from typing import List

class PDFPage:
    def __init__(self, number: int, text: str, image_bytes: bytes):
        self.number = number
        self.text = text
        self.image_bytes = image_bytes

def parse_pdf(pdf_bytes: bytes) -> List[PDFPage]:
    """
    Parses PDF bytes, extracting text and rendering pages as PNG images.
    Returns a list of PDFPage objects.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    
    for i, page in enumerate(doc):
        # Extract text (will be non-empty if the PDF has a selectable text layer)
        text = page.get_text().strip()
        
        # Render the page to a high-quality PNG (150 DPI is standard for OCR)
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
        
        pages.append(PDFPage(number=i + 1, text=text, image_bytes=image_bytes))
        
    return pages
