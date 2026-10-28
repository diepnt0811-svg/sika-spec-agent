# agent_engine.py
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
import config
from knowledge_retriever_tool import EnterpriseKnowledgeRetriever


class SikaAIAgent:
    def __init__(self, vector_store, retriever: EnterpriseKnowledgeRetriever = None):
        self.vector_db = vector_store
        
# Cấu hình Gemini chuẩn (Bỏ api_version để SDK tự nhận diện chuẩn nhất)
        api_key = (
            getattr(config, "GEMINI_API_KEY", None) or os.getenv("GOOGLE_API_KEY")
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            api_version="v1",
            temperature=0.0,
        )

        # Kết nối với Universal Hybrid Retriever Tool
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = EnterpriseKnowledgeRetriever(
                vector_db=vector_store,
                llm_client=self.llm,
                domain_context="Sika Industrial Flooring & Materials"
            )

    def ask_stream(self, question: str):
        print(f"🔍 [Hybrid Retriever & Query Expansion] Đang truy vấn CSDL...")
        
        # Thực hiện tra cứu Lexical + Vector Hybrid Search
        retrieval_results = self.retriever.hybrid_retrieve(query=question, top_k=5)
        static_docs = retrieval_results.get("static_knowledge_results", [])
        
        if not static_docs:
            context_text = "Không tìm thấy tài liệu phù hợp trong CSDL."
            unique_docs = []
        else:
            context_text = "\n\n---\n\n".join(
                [f"TÀI LIỆU: {doc.get('file_name')} (Trang {doc.get('page')}):\n{doc.get('content')}" for doc in static_docs]
            )
            unique_docs = static_docs

        # Self-Consistency System Prompt
        self_consistency_prompt = f"""Bạn là Trợ lý Kỹ thuật Cao cấp của Sika Việt Nam.
Hãy thực hiện quy trình suy luận Self-Consistency qua 3 bước dưới đây để đảm bảo tuyệt đối không bỏ sót thông tin trong CSDL:

--- BỘ TÀI LIỆU TRÍCH XUẤT TỪ CSDL SIKA ---
{context_text}
-----------------------------------------

CÂU HỎI NGƯỜI DÙNG: {question}

QUY TRÌNH THỰC HIỆN:
Bước 1 (Đọc kỹ & Trích xuất): Duyệt qua toàn bộ tài liệu trên, liệt kê TẤT CẢ các sản phẩm/hệ thống có tính chất hoặc khả năng liên quan đến câu hỏi (kể cả các thuật ngữ tương đương như ESD, Dẫn điện, Anti-static, Sàn epoxy/PU...).
Bước 2 (Tự kiểm tra & Đối chiếu): Đánh giá xem có sản phẩm nào bị bỏ sót hay không. Nếu tài liệu có đề cập đến bất kỳ sản phẩm nào phù hợp, KHÔNG ĐƯỢC kết luận là "không có".
Bước 3 (Tổng hợp câu trả lời): Đưa ra câu trả lời chính xác, rõ ràng, nêu rõ tên sản phẩm, đặc tính kỹ thuật và mã tài liệu trích dẫn.

BẮT ĐẦU CÂU TRẢ LỜI TỔNG HỢP CHO NGƯỜI DÙNG:"""

        print("🤖 [Gemini AI] Đang thực thi Self-Consistency & Trả lời...\n")
        
        response_text = ""
        for chunk in self.llm.stream(self_consistency_prompt):
            if isinstance(chunk.content, list):
                text_chunk = "".join([item.get("text", "") for item in chunk.content if isinstance(item, dict) and "text" in item])
            else:
                text_chunk = str(chunk.content)
            
            print(text_chunk, end="", flush=True)
            response_text += text_chunk
            
        print("\n")
        return response_text, unique_docs

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Thay vì gọi trực tiếp ở top-level, anh gọi hàm này bên trong luồng xử lý RAG
embeddings = get_embedding_model()
