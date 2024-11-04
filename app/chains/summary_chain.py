import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
import time

class SummaryChainGroq:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "mixtral-8x7b-32768"
        self.max_tokens_per_chunk = 4000  # Reduced chunk size
        self.max_input_tokens = 3000  # Conservative limit for input
        self.delay_between_calls = 1  # Delay in seconds between API calls
        
    def generate_summary(self, text: str, progress_callback=None) -> str:
        """Generate a comprehensive summary using progressive summarization."""
        # First level: Split into smaller chunks and summarize each
        chunks = self._split_text(text)
        chunk_summaries = []
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress = (i / total_chunks) * 100
                progress_callback(f"Processing chunk {i+1}/{total_chunks} ({progress:.0f}%)")
            
            summary = self._summarize_chunk(chunk)
            if summary:
                chunk_summaries.append(summary)
            time.sleep(self.delay_between_calls)  # Rate limiting
        
        # Second level: Combine chunk summaries if needed
        if len(chunk_summaries) > 1:
            final_summary = self._combine_summaries(chunk_summaries, progress_callback)
        else:
            final_summary = chunk_summaries[0] if chunk_summaries else ""
            
        return final_summary
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into smaller chunks based on sections and paragraphs."""
        # First try to split by sections
        sections = self._split_by_sections(text)
        
        # If sections are still too large, split by paragraphs
        final_chunks = []
        for section in sections:
            if len(section.split()) > self.max_input_tokens:
                paragraphs = self._split_by_paragraphs(section)
                final_chunks.extend(paragraphs)
            else:
                final_chunks.append(section)
        
        return final_chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by common section headers."""
        section_markers = [
            "Abstract", "Introduction", "Methods", "Methodology",
            "Results", "Discussion", "Conclusion", "References"
        ]
        
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            # Check if line might be a section header
            if any(marker.lower() in line.lower() for marker in section_markers) and len(line.split()) < 5:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            current_section.append(line)
            
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into chunks by paragraphs while respecting token limits."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            if current_length + para_length > self.max_input_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
    
    def _summarize_chunk(self, text: str) -> str:
        """Summarize a single chunk of text."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize this section of a research paper, maintaining key details and findings."},
                    {"role": "user", "content": f"Please summarize this section concisely while preserving important information:\n\n{text}"}
                ],
                temperature=0.7,
                max_tokens=1000,  # Reduced output tokens
                top_p=0.9
            )
            
            if completion.choices:
                return completion.choices[0].message.content
            return ""
            
        except Exception as e:
            print(f"Error in summarizing chunk: {str(e)}")
            return ""

    def _combine_summaries(self, summaries: List[str], progress_callback=None) -> str:
        """Progressively combine summaries if needed."""
        if progress_callback:
            progress_callback("Combining summaries...")
            
        try:
            # If combined summaries are still too long, summarize them in batches
            if len(' '.join(summaries)) > self.max_input_tokens:
                batch_size = 3
                batched_summaries = []
                
                for i in range(0, len(summaries), batch_size):
                    batch = summaries[i:i + batch_size]
                    combined_text = "\n\n".join(batch)
                    
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "Create a unified summary from these section summaries of a research paper."},
                            {"role": "user", "content": f"Create a coherent summary from these sections:\n\n{combined_text}"}
                        ],
                        temperature=0.7,
                        max_tokens=1500,
                        top_p=0.9
                    )
                    
                    if completion.choices:
                        batched_summaries.append(completion.choices[0].message.content)
                    time.sleep(self.delay_between_calls)
                
                # Final combination of batch summaries
                return self._combine_summaries(batched_summaries, progress_callback)
            else:
                # Direct combination if within token limits
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Create a final, coherent summary of the entire research paper."},
                        {"role": "user", "content": f"Create a comprehensive two-page summary from these sections:{'---'.join(summaries)}"}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                    top_p=0.9
                )
                
                if completion.choices:
                    return completion.choices[0].message.content
                return ""
                
        except Exception as e:
            print(f"Error in combining summaries: {str(e)}")
            return "\n\n".join(summaries)  # Fallback to simple concatenation