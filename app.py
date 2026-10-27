import os
import sys
import time
import pandas as pd
import streamlit as st
from google import genai

# --- 1. CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title="Sika Spec Life Matrix - AI Specification Agent",
    page_icon="🏗️",
    layout="wide"
)

# --- 2. NHẬP MODULE NỘI BỘ ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

@st.cache_resource
def init_rag_system():
    """Khởi tạo CSDL Vector FAISS và Retriever (cached để tránh load lại nhiều lần)"""
    try:
        from vector_store import VectorDBManager
        from knowledge_retriever_tool import EnterpriseKnowledgeRetriever, init_retriever_tool
        from agent_engine import SikaAIAgent
        from spec_matcher import SpecMatcher

        vdb_manager = VectorDBManager()
        vector_db = vdb_manager.auto_update_or_load()

        if vector_db is not None:
            init_retriever_tool(vector_db=vector_db)
            retriever = EnterpriseKnowledgeRetriever(vector_db=vector_db)
            return retriever
    except Exception as e:
        st.error(f"⚠️ Lỗi khởi tạo RAG Engine: {e}")
    return None

retriever = init_rag_system()

# Khởi tạo Google GenAI Client
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# --- 3. HÀM TRUY VẤN CORE GEMINI AI ---
def ask_sika_agent_core(question: str) -> str:
    if not retriever:
        return "❌ Hệ thống chưa sẵn sàng do chưa nạp được CSDL Vector FAISS."

    if not client:
        return "❌ Chưa cấu hình GEMINI_API_KEY trong Environment Variables / Secrets."

    # 1. Truy vấn tài liệu từ FAISS
    try:
        retrieval_results = retriever.hybrid_retrieve(query=question, top_k=5)
        static_docs = retrieval_results.get("static_knowledge_results", [])
    except Exception as e:
        return f"❌ Lỗi truy vấn FAISS Database: {str(e)}"

    if not static_docs:
        context_text = "Không tìm thấy tài liệu phù hợp trong CSDL."
    else:
        context_text = "\n\n---\n\n".join([
            f"TÀI LIỆU: {doc.get('file_name')} (Trang {doc.get('page')}):\n{doc.get('content')}"
            for doc in static_docs
        ])

    # 2. Xây dựng Prompt Self-Consistency & Guardrails
    prompt = f"""Bạn là Trợ lý Kỹ thuật Cao cấp của Sika Việt Nam.
Hãy thực hiện quy trình suy luận Self-Consistency qua 3 bước dưới đây để đảm bảo tuyệt đối không bỏ sót thông tin trong CSDL:

--- BỘ TÀI LIỆU TRÍCH XUẤT TỪ CSDL SIKA ---
{context_text}
-----------------------------------------

CÂU HỎI / YÊU CẦU NGƯỜI DÙNG: {question}

QUY TRÌNH THỰC HIỆN:
Bước 1 (Đọc kỹ & Trích xuất): Duyệt qua toàn bộ tài liệu trên, liệt kê TẤT CẢ các sản phẩm/hệ thống có tính chất hoặc khả năng liên quan đến câu hỏi.
Bước 2 (Tự kiểm tra & Đối chiếu): Đánh giá xem có sản phẩm nào bị bỏ sót hay không. Nếu CSDL không chứa thông tin, lịch sự báo không tìm thấy. KHÔNG TỰ BỊA ĐẶT TÊN SẢN PHẨM HOẶC THÔNG SỐ.
Bước 3 (Tổng hợp câu trả lời): Đưa ra câu trả lời chính xác, rõ ràng, nêu rõ tên sản phẩm, đặc tính kỹ thuật, độ dày/định mức thi công và mã tài liệu trích dẫn.

BẮT ĐẦU CÂU TRẢ LỜI TỔNG HỢP CHO NGƯỜI DÙNG:"""

    # 3. Danh sách Gemini Model ưu tiên
    model_list = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    response_text = None
    for model_name in model_list:
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                response_text = response.text
                if response_text:
                    break
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "429" in err_msg:
                    time.sleep(1.5)
                else:
                    break
        if response_text:
            break

    if not response_text:
        return "⚠️ Tất cả các mô hình Gemini hiện đang bận hoặc gặp sự cố kết nối. Vui lòng thử lại!"

    if static_docs:
        sources = list(set([f"{d.get('file_name')} (Trang {d.get('page')})" for d in static_docs]))
        response_text += "\n\n📌 **Tài liệu tham chiếu:**\n" + "\n".join([f"- {src}" for src in sources])

    return response_text

# --- 4. HÀM XỬ LÝ ĐỌC FILE ---
def parse_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    content = ""

    try:
        if ext in ['.xlsx', '.xls', '.csv']:
            df = pd.read_csv(uploaded_file) if ext == '.csv' else pd.read_excel(uploaded_file)
            content = f"--- NỘI DUNG FILE BOQ ({filename}) ---\n" + df.to_string(index=False)
        elif ext == '.pdf':
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            content = f"--- NỘI DUNG FILE PDF ({filename}) ---\n"
            for i, page in enumerate(reader.pages):
                content += f"\n[Trang {i+1}]\n" + (page.extract_text() or "")
        elif ext in ['.txt', '.md']:
            content = uploaded_file.read().decode('utf-8')
    except Exception as e:
        content = f"[Lỗi đọc file: {str(e)}]"

    return content

# --- 5. GIAO DIỆN STREAMLIT UI ---
st.title("🏗️ SIKA SPEC LIFE MATRIX - AI SPECIFICATION AGENT")
st.caption("Hệ thống tự động thẩm định & Khớp giải pháp vật liệu Sika cho Dự án Công nghiệp")

# Khởi tạo Chat History trong Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "manual_input_val" not in st.session_state:
    st.session_state.manual_input_val = ""

col1, col2 = st.columns([1, 2])

# --- CỘT TRÁI: ĐẦU VÀO ---
with col1:
    st.subheader("📥 1. Đầu vào Yêu cầu Dự án")
    
    uploaded_file = st.file_uploader(
        "📁 Tải lên File Dự án / BOQ / Specs",
        type=["xlsx", "xls", "csv", "pdf", "txt"]
    )

    manual_input = st.text_area(
        "✍️ Nhập/Bổ sung Yêu cầu Kỹ thuật Thủ công",
        value=st.session_state.manual_input_val,
        placeholder="Ví dụ: Cần sàn Epoxy chống hóa chất cho nhà máy dược GMP, diện tích 2000m2...",
        height=130
    )

    btn_analyze = st.button("⚡ Phân tích & Đề xuất Giải pháp", type="primary", use_container_width=True)

    st.divider()
    st.markdown("💡 **Kịch bản Demo Nhanh:**")
    
    c1, c2, c3 = st.columns(3)
    if c1.button("🧪 Dược GMP"):
        st.session_state.manual_input_val = "Yêu cầu hệ thống sàn hoàn thiện cho nhà máy Dược phẩm GMP Châu Âu: Chịu rửa ma sát cao, kháng hóa chất vệ sinh, bề mặt phẳng mịn nhẵn bóng, độ dầy 3mm-4mm, tiêu chuẩn xả thải VOC thấp."
        st.rerun()
    if c2.button("🏭 Bán dẫn ESD"):
        st.session_state.manual_input_val = "Dự án nhà máy đóng gói Bán dẫn SHTP: Yêu cầu hệ thống sàn ESD kiểm soát tĩnh điện (10^6 - 10^9 Ohms), chịu tải nhẹ đến trung bình, không phát sinh bụi sạch Class 100."
        st.rerun()
    if c3.button("💧 Nước thải"):
        st.session_state.manual_input_val = "Hạng mục bể chứa nước thải công nghiệp pH 2-12: Yêu cầu vật liệu bọc phủ chống thấm và kháng mài mòn sinh học, thi công được trên bề mặt bê tông ẩm."
        st.rerun()

# Xử lý khi nhấn nút Phân tích
if btn_analyze:
    file_content = parse_uploaded_file(uploaded_file) if uploaded_file else ""
    combined_query = ""
    if file_content:
        combined_query += f"Dữ liệu từ File dự án/BOQ:\n{file_content}\n\n"
    if manual_input.strip():
        combined_query += f"Yêu cầu bổ sung từ người dùng:\n{manual_prompt.strip()}" if 'manual_prompt' in locals() else f"Yêu cầu bổ sung:\n{manual_input.strip()}"

    if not combined_query.strip():
        st.warning("⚠️ Vui lòng tải lên file BOQ/Dự án hoặc nhập yêu cầu kỹ thuật.")
    else:
        with st.spinner("🔍 Đang truy vấn CSDL FAISS Vector & Gemini LLM..."):
            start_time = time.time()
            res = ask_sika_agent_core(combined_query)
            elapsed = round(time.time() - start_time, 2)
            
            user_msg = manual_input.strip() if manual_input.strip() else "Phân tích file BOQ dự án"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            st.session_state.messages.append({"role": "assistant", "content": f"⏱️ *Xử lý trong {elapsed}s*\n\n" + res})
            st.session_state.last_result = res

# --- CỘT PHẢI: KHUNG CHAT & KẾT QUẢ ---
with col2:
    st.subheader("📊 2. Kết quả Phân tích & Hỏi đáp Kỹ thuật")

    # Hiển thị lịch sử Chat
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Ô nhập câu hỏi thảo luận thêm
    if user_followup := st.chat_input("💬 Hỏi đáp thêm về kết quả (độ dày, định mức, quy trình thi công...)..."):
        st.session_state.messages.append({"role": "user", "content": user_followup})
        
        contextual_prompt = (
            f"Dựa trên kết quả thẩm định giải pháp Sika trước đó:\n\n{st.session_state.last_result}\n\n"
            f"Người dùng hỏi thêm: {user_followup}"
        )
        
        with st.spinner("🤖 Sika Agent đang suy luận..."):
            start_time = time.time()
            bot_res = ask_sika_agent_core(contextual_prompt)
            elapsed = round(time.time() - start_time, 2)
            
            st.session_state.messages.append({"role": "assistant", "content": f"⏱️ *Phản hồi trong {elapsed}s*\n\n" + bot_res})
        st.rerun()
