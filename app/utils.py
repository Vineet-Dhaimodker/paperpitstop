from typing import Dict, List
import re

def format_summary(summary: str) -> str:
    """Format the generated summary for better readability."""
    # Remove any generated prompt text
    summary = re.sub(r'^Summary:|^Summary of the paper:', '', summary, flags=re.IGNORECASE)
    
    # Clean up whitespace
    summary = summary.strip()
    summary = re.sub(r'\n\s*\n', '\n\n', summary)
    
    return summary

def format_key_points(points: List[str]) -> str:
    """Format key points as bullet points."""
    return "\n".join(f"• {point}" for point in points)

def validate_pdf(file_bytes: bytes) -> bool:
    """Validate if the uploaded file is a valid PDF."""
    return file_bytes.startswith(b'%PDF')

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to specified length while maintaining complete sentences."""
    if len(text) <= max_length:
        return text
        
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period != -1:
        return truncated[:last_period + 1]
    return truncated