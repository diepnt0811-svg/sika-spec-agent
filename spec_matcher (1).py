# spec_matcher.py
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from knowledge_retriever_tool import EnterpriseKnowledgeRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpecMatcher")


class SpecMatcher:
    def __init__(
        self,
        retriever: EnterpriseKnowledgeRetriever,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        self.retriever = retriever
        if llm:
            self.llm = llm
        else:
            api_key = getattr(config, "GEMINI_API_KEY", None) or os.getenv(
                "GOOGLE_API_KEY"
            )
            # Dùng model gemini-1.5-flash với v1beta
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1,
            )

    def _analyze_single_item(
        self, item_description: str, project_context_str: str = ""
    ) -> Dict[str, Any]:
        retrieval_results = self.retriever.hybrid_retrieve(
            query=item_description, top_k=2
        )
        static_docs = retrieval_results.get("static_knowledge_results", [])

        if not static_docs:
            return {
                "boq_item": item_description,
                "recommended_sika_product": "Khớp nối thất bại / Cần kiểm tra lại",
                "match_score": "0%",
                "technical_justification": "Không tìm thấy dữ liệu sản phẩm tương thích trong CSDL FAISS.",
                "referenced_docs": [],
            }

        matched_context = "\n\n".join(
            [
                f"--- TÀI LIỆU: {doc.get('file_name')} (Trang {doc.get('page')}) ---\n{doc.get('content')}"
                for doc in static_docs
            ]
        )

        matching_prompt = f"""Bạn là Chuyên gia Tư vấn Kỹ thuật Giải pháp Sika.
Hãy phân tích hạng mục vật liệu/yêu cầu kỹ thuật từ dự án dưới đây và chọn ra giải pháp/sản phẩm Sika tối ưu nhất dựa TRỰC TIẾP trên CSDL tài liệu kỹ thuật được cung cấp.

--- MÔ TẢ HẠNG MỤC BOQ DỰ ÁN ---
{item_description}
{f"Ngữ cảnh dự án chung: {project_context_str}" if project_context_str else ""}

--- TÀI LIỆU KỸ THUẬT SIKA TRÍCH XUẤT TỪ CSDL ---
{matched_context}

--- YÊU CẦU ĐẦU RA ---
Trả về kết quả duy nhất dưới định dạng JSON (không thêm lời dẫn hay ký tự markdown ```json):
{{
  "recommended_sika_product": "Tên thương mại sản phẩm Sika tối ưu (Ví dụ: Sikafloor-264, Sika Ucrete MF, Sikadur-31...)",
  "match_score": "Phần trăm độ tương thích (Ví dụ: 95%)",
  "technical_justification": "Lý do kỹ thuật giải thích ngắn gọn lý do chọn sản phẩm này.",
  "system_thickness": "Độ dày khuyến nghị hoặc định mức thi công (nếu không có ghi N/A)"
}}
"""
        try:
            response = self.llm.invoke(matching_prompt)
            content = getattr(response, "content", str(response)).strip()
            
            # Trích xuất JSON bằng Regex an toàn
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            parsed_result = json.loads(content.strip())
            parsed_result["boq_item"] = item_description
            parsed_result["referenced_docs"] = [
                f"{d.get('file_name')} (P.{d.get('page')})" for d in static_docs
            ]
            return parsed_result
        except Exception as e:
            logger.error(
                f"Lỗi khi khớp nối hạng mục '{item_description[:30]}...': {e}"
            )
            return {
                "boq_item": item_description,
                "recommended_sika_product": "Lỗi phân tích",
                "match_score": "N/A",
                "technical_justification": f"Chi tiết lỗi: {str(e)}",
                "referenced_docs": [],
            }

    def match_boq_data(
        self,
        project_context: Dict[str, Any],
        description_column_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        description_keywords = description_column_keywords or [
            "mô tả",
            "description",
            "nội dung",
            "vật liệu",
            "quy cách",
            "spec",
            "item",
            "hạng mục",
        ]
        project_files = project_context.get("project_files", [])
        session_prompt = project_context.get("session_prompt", "")

        # 1. Bộ lọc thông minh tiền xử lý
        valid_construction_keywords = [
            "sàn", "floor", "bê tông", "concrete", "chống thấm", "waterproof", 
            "epoxy", "pu", "ucrete", "sơn", "paint", "vữa", "grout", "trám", 
            "sealant", "khe", "joint", "gia cố", "strengthening", "mặt cầu", "màng"
        ]

        items_to_process = []

        for file_info in project_files:
            file_type = file_info.get("file_type", "")

            if file_type == "excel" and "content" in file_info:
                sheets_data = file_info["content"]
                for sheet_name, sheet_content in sheets_data.items():
                    records = sheet_content.get("records", [])
                    for idx, row in enumerate(records):
                        if not isinstance(row, dict):
                            continue
                        desc_text = ""
                        for col_key, col_val in row.items():
                            if any(
                                kw in str(col_key).lower()
                                for kw in description_keywords
                            ):
                                if col_val and str(col_val).strip():
                                    desc_text += f"{col_key}: {col_val}; "
                        if not desc_text:
                            str_vals = [
                                str(v).strip()
                                for v in row.values()
                                if v and len(str(v).strip()) > 10
                            ]
                            if str_vals:
                                desc_text = " ".join(str_vals)

                        clean_desc = desc_text.strip().lower()
                        if clean_desc and len(clean_desc) > 5:
                            if any(kw in clean_desc for kw in valid_construction_keywords):
                                items_to_process.append(desc_text.strip())

        items_to_process = items_to_process[:15]

        # 2. Xử lý đa luồng song song
        matched_summary = [None] * len(items_to_process)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {
                executor.submit(self._analyze_single_item, desc, session_prompt): i
                for i, desc in enumerate(items_to_process)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    matched_summary[index] = future.result()
                except Exception as exc:
                    logger.error(f"Hạng mục {index} gặp lỗi: {exc}")
                    matched_summary[index] = {
                        "boq_item": items_to_process[index],
                        "recommended_sika_product": "Lỗi xử lý luồng",
                        "match_score": "N/A",
                        "technical_justification": str(exc),
                        "referenced_docs": [],
                    }

        return [res for res in matched_summary if res is not None]

    def export_to_dataframe(
        self, matched_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        formatted_list = []
        for res in matched_results:
            formatted_list.append(
                {
                    "Hạng Mục BOQ / Yêu Cầu": res.get("boq_item", ""),
                    "Giải Pháp Sika Đề Xuất": res.get(
                        "recommended_sika_product", ""
                    ),
                    "Độ Tương Thích": res.get("match_score", ""),
                    "Độ Dày / Định Mức": res.get("system_thickness", "N/A"),
                    "Lý Do Kỹ Thuật": res.get("technical_justification", ""),
                    "Tài Liệu Trích Dẫn": ", ".join(
                        res.get("referenced_docs", [])
                    ),
                }
            )
        return pd.DataFrame(formatted_list)
