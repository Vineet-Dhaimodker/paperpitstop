import PyPDF2
import io
from typing import BinaryIO

class PDFProcessor:
    def __init__(self):
        self.max_pages = 30  # Limit processing to first 30 pages
        
    def process_pdf(self, file: BinaryIO) -> str:
        """Process PDF file and return text content."""
        try:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from pages
            text_content = []
            for page_num in range(min(len(pdf_reader.pages), self.max_pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
            
            return "\n".join(text_content)
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing unnecessary whitespace and characters."""
        # Remove multiple newlines
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
        
        # Remove unusual characters
        text = "".join(char for char in text if ord(char) < 128)
        
        return text