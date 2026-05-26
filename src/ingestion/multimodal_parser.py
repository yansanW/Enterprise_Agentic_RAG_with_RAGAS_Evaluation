# src/ingestion/multimodal_parser.py
import os
import fitz  # PyMuPDF (pip install pymupdf)
from langchain_core.documents import Document
from src import config
# Note: In production, you would import your local VLM client or ChatGoogleGenerativeAI here

class MultimodalParser:
    def __init__(self, vlm_client=None):
        """Initializes the layout extractor and assigns the VLM engine."""
        self.vlm = vlm_client  # Can be your local server model or an API client

    def _extract_page_as_image(self, doc, page_num: int) -> str:
        """Converts a single PDF page into a temporary PNG image for visual analysis."""
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)  # Standard resolution for structural processing
        
        temp_image_path = f"data/raw_docs/page_{page_num + 1}.png"
        pix.save(temp_image_path)
        return temp_image_path

    def parse_document(self, file_path: str) -> list[Document]:
        """
        Processes a PDF document multimodally by analyzing the layout structure
        and using a VLM to transcribe complex tables and images.
        """
        # 1. Enforce strict parameter validation rules
        if not file_path or str(file_path).strip() == "":
            raise ValueError("File path cannot be empty.")
        
        print(f"Opening document for multimodal analysis: {file_path}")
        doc = fitz.open(file_path)
        processed_chunks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # 1. Structural Layer: Identify tables using PyMuPDF's built-in table finder
            tables = page.find_tables()
            
            # 2. Extract standard clear text sections normally
            text_content = page.get_text("text").strip()
            
            if tables:
                print(f" Found {len(tables.tables)} structural tables on Page {page_num + 1}")
                # Convert the page to an image so the VLM can analyze the table context visually
                page_img = self._extract_page_as_image(doc, page_num)
                
                # Mock prompt simulating passing the image crop to your VLM model
                vlm_prompt = (
                    "Look at the table in this page image. Convert it accurately into a "
                    "clean Markdown table format with proper header columns. Do not include conversational text."
                )
                
                # Here you would call: markdown_table = self.vlm.invoke(vlm_prompt, page_img)
                markdown_table = "| Metric | Value |\n|---|---|\n| Sample Extracted Metric | 100% |"
                
                # Append the structured markdown table chunk to our outputs array
                table_metadata = {"source": file_path, "page": page_num + 1, "type": "table"}
                processed_chunks.append(Document(page_content=markdown_table, metadata=table_metadata))
                
                # Clean up the temporary image asset
                if os.path.exists(page_img):
                    os.remove(page_img)

            # 3. Handle standard paragraph chunks
            if text_content:
                metadata = {"source": file_path, "page": page_num + 1, "type": "text"}
                processed_chunks.append(Document(page_content=text_content, metadata=metadata))

        return processed_chunks

if __name__ == "__main__":
    # Quick sanity verification routine
    parser = MultimodalParser()
    # You can drop a multi-column or table-heavy research paper into your data block to test
    # chunks = parser.parse_document("data/raw_docs/your_research_paper.pdf")