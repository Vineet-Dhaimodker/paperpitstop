import os
from typing import Dict
from dotenv import load_dotenv
from groq import Groq

class ExtractionChainGroq:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "mixtral-8x7b-32768"
    
    def extract_info(self, text: str) -> Dict[str, str]:
        """Extract key information from the research paper."""
        # Create system message for extraction
        system_message = """You are a research paper analysis assistant. For each aspect requested, 
        provide clear, structured information extracted from the paper. Focus on accuracy and relevance."""
        
        # Extract information for each aspect
        contributions = self._generate_response(text, "main contributions", system_message)
        methodology = self._generate_response(text, "methodology", system_message)
        results = self._generate_response(text, "results", system_message)
        
        return {
            "contributions": contributions,
            "methodology": methodology,
            "results": results
        }
    
    def _generate_response(self, text: str, aspect: str, system_message: str) -> str:
        """Generate response using the Groq API."""
        prompt = self._create_extraction_prompt(text, aspect)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        )
        
        if completion.choices:
            return completion.choices[0].message.content
        return ""
        
    def _create_extraction_prompt(self, text: str, aspect: str) -> str:
        """Create a prompt for extracting specific information."""
        return f"""From the following research paper, please extract and summarize the {aspect}. 
Focus on providing a clear and concise explanation:

{text}

Please provide a detailed summary of the {aspect}:"""