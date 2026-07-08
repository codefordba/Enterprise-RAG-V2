# src/processing/layout_parser.py
import logging
import pdfplumber
from typing import List, Dict, Any

logger = logging.getLogger("LayoutParser")

class LayoutAwareParser:
    @staticmethod
    def extract_elements(file_source: Any) -> List[Dict[str, Any]]:
        """Deconstructs high-density PDFs into isolated text paragraphs and formatted markdown tables."""
        extracted_elements = []
        
        try:
            with pdfplumber.open(file_source) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    # 1. Isolate and parse embedded table matrices
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            cleaned_rows = [[str(cell or "").strip() for cell in row] for row in table]
                            if len(cleaned_rows) > 0:
                                headers = cleaned_rows[0]
                                m_str = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
                                for row in cleaned_rows[1:]:
                                    m_str += f"| {' | '.join(row)} |\n"
                                
                                extracted_elements.append({
                                    "content": f"[Table Layout Structure]:\n{m_str}",
                                    "type": "table",
                                    "page": page_idx
                                })
                    
                    # 2. Extract standard running text
                    prose_text = page.extract_text()
                    if prose_text:
                        extracted_elements.append({
                            "content": prose_text,
                            "type": "prose",
                            "page": page_idx
                        })
            return extracted_elements
        except Exception as e:
            logger.error(f"Error parsing layout elements: {str(e)}")
            raise e