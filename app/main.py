import streamlit as st
import os
from processor import PDFProcessor
from chains.summary_chain import SummaryChainGroq
from chains.extraction_chain import ExtractionChainGroq
from utils import format_summary
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    st.set_page_config(page_title="PaperPitStop", layout="wide")
    st.title("PaperPitStop - Research Paper Summarizer (Powered by Groq)")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
    
    if uploaded_file:
        # Create a placeholder for the progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message):
            """Update progress bar and status message."""
            if "%" in message:
                try:
                    progress = int(message.split("(")[1].split("%")[0])
                    progress_bar.progress(progress / 100)
                except:
                    pass
            status_text.text(message)
        
        try:
            # Process PDF
            with st.spinner("Processing PDF..."):
                processor = PDFProcessor()
                text_content = processor.process_pdf(uploaded_file)
                
                # Initialize chains
                summary_chain = SummaryChainGroq()
                extraction_chain = ExtractionChainGroq()
                
                # Generate summary and extract key information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Executive Summary")
                    summary = summary_chain.generate_summary(text_content, update_progress)
                    st.write(summary)
                
                with col2:
                    st.subheader("Key Information")
                    key_info = extraction_chain.extract_info(text_content)
                    
                    st.write("### Main Contributions")
                    st.write(key_info["contributions"])
                    
                    st.write("### Methodology")
                    st.write(key_info["methodology"])
                    
                    st.write("### Results")
                    st.write(key_info["results"])
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()