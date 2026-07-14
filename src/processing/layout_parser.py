# src/processing/layout_parser.py
import logging
import pdfplumber
import base64
import requests
from io import BytesIO
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import Config

logger = logging.getLogger("LayoutParser")

class LayoutAwareParser:
    @staticmethod
    def _describe_image_via_gemini(image_base64: str) -> str:
        """Calls the OpenAI-compatible multimodal endpoint of Gemini 3.5 Flash to describe an image."""
        url = f"{Config.LLM_API_BASE_URL}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.LLM_API_KEY}"
        }
        
        payload = {
            "model": "gemini-3.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Identify if this is a chart, diagram, table, or photograph, and generate a detailed textual description of it. "
                                "Focus on extracting all raw numbers, dates, titles, data points, labels, and trends. "
                                "For charts, explain what the axes represent and describe any clear trends or anomalies. "
                                "Return only the factual description. Be precise and thorough."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            res_data = response.json()
            description = res_data["choices"][0]["message"]["content"].strip()
            return description
        except Exception as e:
            logger.error(f"Multimodal LLM description request failed: {str(e)}")
            return "[Visual Element: Description generation failed due to api timeout or error]"

    @staticmethod
    def _format_table_to_markdown(table: List[List[Any]]) -> str:
        """Converts a grid matrix into clean Markdown table formatting."""
        if not table or not table[0]:
            return ""
        
        cleaned_rows = [[str(cell or "").strip() for cell in row] for row in table]
        headers = cleaned_rows[0]
        
        # Guard against completely empty headers
        if not any(headers):
            headers = [f"Col {i+1}" for i in range(len(headers))]
            
        header_row = f"| {' | '.join(headers)} |"
        separator_row = f"| {' | '.join(['---'] * len(headers))} |"
        
        body_rows = []
        for row in cleaned_rows[1:]:
            # Ensure row length matches header length
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            body_rows.append(f"| {' | '.join(row)} |")
            
        return f"\n{header_row}\n{separator_row}\n" + "\n".join(body_rows) + "\n"

    @staticmethod
    def _is_inside_bboxes(bbox: tuple, table_bboxes: List[tuple]) -> bool:
        """Determines if a word bbox center is inside any table bounding box."""
        x0, top, x1, bottom = bbox
        cx = (x0 + x1) / 2
        cy = (top + bottom) / 2
        for tx0, ttop, tx1, tbottom in table_bboxes:
            if tx0 <= cx <= tx1 and ttop <= cy <= tbottom:
                return True
        return False

    @classmethod
    def extract_elements(cls, file_source: Any) -> List[Dict[str, Any]]:
        """Deconstructs high-density PDFs into structural Markdown (prose, headings, tables, image descriptions)."""
        extracted_elements = []
        
        try:
            with pdfplumber.open(file_source) as pdf:
                # First, analyze the whole document to determine the dominant body text font size
                all_sizes = []
                for page in pdf.pages:
                    words = page.extract_words(extra_attrs=["size"])
                    all_sizes.extend([w["size"] for w in words if "size" in w])
                
                # Compute median size in pure python
                if all_sizes:
                    sorted_sizes = sorted(all_sizes)
                    body_font_size = sorted_sizes[len(sorted_sizes) // 2]
                else:
                    body_font_size = 10.0
                
                logger.info(f"Detected dominant document body font size: {body_font_size}")
                
                # We will collect image description tasks to run them in parallel
                image_tasks = []
                
                for page_idx, page in enumerate(pdf.pages, start=1):
                    page_width = page.width
                    page_height = page.height
                    
                    # 1. Detect Tables and locate their bounding boxes
                    tables = page.find_tables()
                    table_bboxes = [t.bbox for t in tables]
                    
                    # 2. Render and Crop valid images/charts
                    page_images = page.images
                    valid_images = []
                    for img in page_images:
                        w = img["x1"] - img["x0"]
                        h = img["bottom"] - img["top"]
                        # Filter out tiny shapes, icons, and page-wide background decorative patterns
                        if w < 40 or h < 40:
                            continue
                        if w >= 0.95 * page_width and h >= 0.95 * page_height:
                            continue
                        valid_images.append(img)
                    
                    # 3. Extract words excluding those inside tables
                    raw_words = page.extract_words(extra_attrs=["fontname", "size"])
                    non_table_words = [
                        w for w in raw_words
                        if not cls._is_inside_bboxes((w["x0"], w["top"], w["x1"], w["bottom"]), table_bboxes)
                    ]
                    
                    # Group words into lines
                    lines = []
                    sorted_words = sorted(non_table_words, key=lambda w: (w["top"], w["x0"]))
                    for word in sorted_words:
                        if not lines or abs(word["top"] - lines[-1]["top"]) > 3.0:
                            lines.append({
                                "top": word["top"],
                                "bottom": word["bottom"],
                                "words": [word]
                            })
                        else:
                            lines[-1]["words"].append(word)
                            lines[-1]["bottom"] = max(lines[-1]["bottom"], word["bottom"])
                    
                    # Process lines into paragraph/heading structural blocks
                    layout_blocks = []
                    
                    # Construct text lines
                    for line in lines:
                        line_text = " ".join([w["text"] for w in line["words"]]).strip()
                        if not line_text:
                            continue
                            
                        # Average font size in the line
                        avg_size = sum([w["size"] for w in line["words"]]) / len(line["words"])
                        
                        # Heading detection rules
                        if avg_size >= body_font_size + 3.0 or avg_size >= 14.0:
                            heading_type = "heading_1"
                            formatted_text = f"\n# {line_text}\n"
                        elif avg_size >= body_font_size + 1.2 or avg_size >= 12.0:
                            heading_type = "heading_2"
                            formatted_text = f"\n## {line_text}\n"
                        else:
                            heading_type = "prose"
                            # Standardize lists
                            lower_text = line_text.lower()
                            if line_text.startswith(("- ", "* ", "• ", "o ")):
                                formatted_text = f"- {line_text[2:].strip()}"
                            elif any(line_text.startswith(f"{i}. ") for i in range(1, 20)):
                                formatted_text = line_text
                            else:
                                formatted_text = line_text
                        
                        layout_blocks.append({
                            "top": line["top"],
                            "content": formatted_text,
                            "type": heading_type
                        })
                    
                    # Add tables as layout blocks
                    for table in tables:
                        md_table = cls._format_table_to_markdown(table.extract())
                        if md_table:
                            layout_blocks.append({
                                "top": table.bbox[1], # top coordinate
                                "content": md_table,
                                "type": "table"
                            })
                    
                    # Add images as layout blocks (temporarily containing placeholder, to be updated after multimodal processing)
                    for img_idx, img in enumerate(valid_images):
                        x0 = max(0, min(img["x0"], page_width))
                        top = max(0, min(img["top"], page_height))
                        x1 = max(0, min(img["x1"], page_width))
                        bottom = max(0, min(img["bottom"], page_height))
                        
                        if (x1 - x0) > 5 and (bottom - top) > 5:
                            try:
                                cropped = page.crop((x0, top, x1, bottom))
                                pil_img = cropped.to_image(resolution=150).original
                                buffered = BytesIO()
                                pil_img.save(buffered, format="PNG")
                                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                
                                placeholder_key = f"__IMAGE_DESC_PAGE_{page_idx}_IMG_{img_idx}__"
                                layout_blocks.append({
                                    "top": top,
                                    "content": placeholder_key,
                                    "type": "image",
                                    "image_b64": img_b64,
                                    "placeholder": placeholder_key
                                })
                            except Exception as crop_err:
                                logger.warning(f"Failed to crop/render image bounding box on page {page_idx}: {str(crop_err)}")
                    
                    # Sort layout blocks on the page by vertical reading order (top coordinate)
                    layout_blocks.sort(key=lambda b: b["top"])
                    
                    # Combine consecutive prose lines into paragraphs to ensure cohesive markdown structure
                    combined_blocks = []
                    current_prose = []
                    
                    for block in layout_blocks:
                        if block["type"] == "prose":
                            content = block["content"]
                            if content.startswith("- ") or any(content.startswith(f"{i}. ") for i in range(1, 20)):
                                # If it's a list item, flush current prose first
                                if current_prose:
                                    combined_blocks.append({
                                        "content": "\n".join(current_prose) + "\n",
                                        "type": "prose"
                                    })
                                    current_prose = []
                                combined_blocks.append({
                                    "content": content,
                                    "type": "list"
                                })
                            else:
                                current_prose.append(content)
                        else:
                            # Flush prose before table, heading, or image
                            if current_prose:
                                combined_blocks.append({
                                    "content": "\n".join(current_prose) + "\n",
                                    "type": "prose"
                                })
                                current_prose = []
                            combined_blocks.append(block)
                            
                    if current_prose:
                        combined_blocks.append({
                            "content": "\n".join(current_prose) + "\n",
                            "type": "prose"
                        })
                    
                    # Build page markdown and collect image tasks
                    page_md_segments = []
                    for cb in combined_blocks:
                        if cb["type"] == "image":
                            # Queue image description task
                            image_tasks.append({
                                "placeholder": cb["placeholder"],
                                "image_b64": cb["image_b64"],
                                "page_idx": page_idx
                            })
                            page_md_segments.append(cb["placeholder"])
                        else:
                            page_md_segments.append(cb["content"])
                            
                    # Assemble page-level text
                    page_md = "\n".join(page_md_segments).strip()
                    
                    # Add explicit Page Breaks as requested
                    page_break_marker = f"\n\n---\n<!-- PAGE BREAK Page {page_idx} -->\n\n"
                    
                    extracted_elements.append({
                        "content": page_md,
                        "type": "page_markdown",
                        "page": page_idx,
                        "page_break_marker": page_break_marker
                    })
                
                # 4. Resolve Image/Chart Descriptions asynchronously via ThreadPoolExecutor
                if image_tasks:
                    logger.info(f"Found {len(image_tasks)} image/chart extraction areas. Querying Gemini Vision API in parallel...")
                    descriptions = {}
                    
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        future_to_task = {
                            executor.submit(cls._describe_image_via_gemini, task["image_b64"]): task
                            for task in image_tasks
                        }
                        for future in as_completed(future_to_task):
                            task = future_to_task[future]
                            try:
                                desc = future.result()
                                # Format visual description nicely as a section in the document context
                                formatted_desc = f"\n\n[Chart/Image Description (Page {task['page_idx']}):\n{desc}]\n\n"
                                descriptions[task["placeholder"]] = formatted_desc
                            except Exception as ex:
                                logger.error(f"Async image description resolution failed: {str(ex)}")
                                descriptions[task["placeholder"]] = "\n\n[Visual Element: Failed to generate multimodal description]\n\n"
                    
                    # Inject resolved descriptions back into the document page markdown
                    for element in extracted_elements:
                        for placeholder, desc_text in descriptions.items():
                            if placeholder in element["content"]:
                                element["content"] = element["content"].replace(placeholder, desc_text)
                                
            return extracted_elements
        except Exception as e:
            logger.error(f"Error parsing layout elements: {str(e)}")
            raise e