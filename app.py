import os
import sys
import time
import pandas as pd
import streamlit as st

# --- 1. CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title="Sika Spec Life Matrix - AI Specification Agent",
    page_icon="🏗️",
    layout="wide"
)

# --- 2. KIỂM TRA & NHẬP THƯ VIỆN GOOGLE GENAI ---
try:
    from google import genai
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False
    st.error(
        "⚠️ Thiếu thư viện `google-genai`. Anh vui lòng thêm `google-genai` vào file `requirements.txt` "
        "trên GitHub và chọn 'Reboot app' trên Streamlit Cloud."
    )

# --- 3. NHẬP MODULE NỘI BỘ (RAG ENGINE & FAISS) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import traceback

@st.cache_resource
def init_rag_system():
    """Khởi tạo CSDL Vector FAISS và Retriever với cơ chế debug chi tiết"""
    try:
        # Đảm bảo sys.path chứa thư mục chứa app.py
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)

        from vector_store import VectorDBManager
        from knowledge_retriever_tool import EnterpriseKnowledgeRetriever, init_retriever_tool

        vdb_manager = VectorDBManager()
        
        # Gọi nạp CSDL
        vector_db = vdb_manager.auto_update_or_load()

        if vector_db is not None:
            init_retriever_tool(vector_db=vector_db)
            retriever = EnterpriseKnowledgeRetriever(vector_db=vector_db)
            st.success("✅ Đã nạp thành công CSDL FAISS Vector DB!")
            return retriever
        else:
            st.error("⚠️ `vector_db` trả về `None`. Lỗi có thể do không tìm thấy file `index.faiss` / `index.pkl` hoặc lỗi nạp Embedding Model.")
    except Exception as e:
        st.error(f"❌ Lỗi chi tiết khi khởi tạo RAG Engine: {str(e)}")
        st.code(traceback.format_exc(), language="python")
    return None

retriever = init_rag_system()

# --- KHỞI TẠO CLIENT VỚI HỖ TRỢ CẢ 2 ĐỊNH DẠNG KEY (AIzaSy... & AQ.Ab...) ---
raw_api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    raw_api_key = str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
else:
    raw_api_key = str(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

client = None
if HAS_GENAI_SDK and raw_api_key:
    try:
        if raw_api_key.startswith("AQ."):
            # Xử lý định dạng Express Key / Bearer Token
            from google.genai import types
            client = genai.Client(
                http_options=types.HttpOptions(
                    headers={"Authorization": f"Bearer {raw_api_key}"}
                )
            )
        else:
            # Xử lý định dạng Standard API Key
            client = genai.Client(api_key=raw_api_key)
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo Gemini Client: {str(e)}")

# --- 4. HÀM TRUY VẤN CORE GEMINI AI ---
import os
import time
import streamlit as st
import google.generativeai as genai_legacy

# --- CẤU HÌNH API KEY VỚI HỖ TRỢ GOOGLE-GENERATIVEAI ---
raw_api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    raw_api_key = str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
else:
    raw_api_key = str(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

if raw_api_key:
    # Cấu hình SDK chuẩn
    genai_legacy.configure(api_key=raw_api_key)


def ask_sika_agent_core(question: str) -> str:
    if not raw_api_key:
        return "❌ Chưa cấu hình API Key. Anh hãy bổ sung `GEMINI_API_KEY` vào mục **Secrets** trên Streamlit Cloud."[cite: 7]

    if not retriever:
        return "❌ Hệ thống chưa sẵn sàng do chưa kết nối được CSDL Vector FAISS."

    # 1. Truy vấn tài liệu từ CSDL FAISS
    try:
        retrieval_results = retriever.hybrid_retrieve(query=question, top_k=5)
        static_docs = retrieval_results.get("static_knowledge_results", [])
    except Exception as e:
        return f"❌ Lỗi truy vấn CSDL FAISS: {str(e)}"

    if not static_docs:
        context_text = "Không tìm thấy tài liệu phù hợp trực tiếp trong CSDL."
    else:
        context_text = "\n\n---\n\n".join([
            f"TÀI LIỆU: {doc.get('file_name')} (Trang {doc.get('page')}):\n{doc.get('content')}"
            for doc in static_docs
        ])

    # 2. Xây dựng Prompt Self-Consistency
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

    # 3. Danh sách model theo cú pháp google-generativeai
    model_list = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-exp"
    ]

    response_text = None
    last_error_details = []

    for model_name in model_list:
        try:
            model = genai_legacy.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                response_text = response.text
                break
        except Exception as e:
            last_error_details.append(f"Model [{model_name}]: {str(e)}")

    if not response_text:
        error_summary = "\n".join(last_error_details)
        return f"⚠️ **Lỗi xác thực Google API:**\n```text\n{error_summary}\n```\n*Anh vui lòng kiểm tra lại Key trên Google AI Studio hoặc Reboot app trên Streamlit Cloud.*"

    if static_docs:
        sources = list(set([f"{d.get('file_name')} (Trang {d.get('page')})" for d in static_docs]))
        response_text += "\n\n📌 **Tài liệu tham chiếu:**\n" + "\n".join([f"- {src}" for src in sources])

    return response_text

# --- 5. HÀM ĐỌC FILE TẢI LÊN ---
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

# --- 6. GIAO DIỆN STREAMLIT UI ---
st.title("🏗️ SIKA SPEC LIFE MATRIX - AI SPECIFICATION AGENT")
st.caption("Hệ thống tự động thẩm định & Khớp giải pháp vật liệu Sika cho Dự án Công nghiệp")

# Quản lý State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "manual_input_val" not in st.session_state:
    st.session_state.manual_input_val = ""

col_input, col_chat = st.columns([1, 2])

# --- CỘT TRÁI: ĐẦU VÀO YÊU CẦU ---
with col_input:
    st.subheader("📥 1. Đầu vào Yêu cầu Dự án")
    
    uploaded_file = st.file_uploader(
        "📁 Tải lên File Dự án / BOQ / Specs",
        type=["xlsx", "xls", "csv", "pdf", "txt"]
    )

    manual_input = st.text_area(
        "✍️ Nhập/Bổ sung Yêu cầu Kỹ thuật Thủ công",
        value=st.session_state.manual_input_val,
        placeholder="Ví dụ: Cần sàn Epoxy chống hóa chất cho nhà máy dược GMP, diện tích 2000m2...",
        height=140
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

# Xử lý sự kiện Phân tích
if btn_analyze:
    file_content = parse_uploaded_file(uploaded_file) if uploaded_file else ""
    combined_query = ""
    if file_content:
        combined_query += f"Dữ liệu từ File dự án/BOQ:\n{file_content}\n\n"
    if manual_input.strip():
        combined_query += f"Yêu cầu bổ sung:\n{manual_input.strip()}"

    if not combined_query.strip():
        st.warning("⚠️ Vui lòng tải lên file BOQ/Dự án hoặc nhập yêu cầu kỹ thuật.")
    else:
        with st.spinner("🔍 Đang đối chiếu CSDL FAISS Vector & Gemini LLM..."):
            start_time = time.time()
            res = ask_sika_agent_core(combined_query)
            elapsed = round(time.time() - start_time, 2)
            
            user_msg = manual_input.strip() if manual_input.strip() else "Phân tích file dự án/BOQ"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            st.session_state.messages.append({"role": "assistant", "content": f"⏱️ *Thời gian xử lý: {elapsed}s*\n\n" + res})
            st.session_state.last_result = res

# --- CỘT PHẢI: HIỂN THỊ KẾT QUẢ & CHAT THẢO LUẬN ---
with col_chat:
    st.subheader("📊 2. Kết quả Phân tích & Thảo luận Kỹ thuật")

    # Vùng chứa khung hội thoại
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Nhập câu hỏi phụ
    if user_followup := st.chat_input("💬 Hỏi đáp thêm (định mức thi công, tiêu chuẩn ASTM, so sánh giải pháp...)..."):
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
