import os
import time
import pandas as pd
import streamlit as st
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & API KEY BẢO MẬT
# ==========================================
st.set_page_config(
    page_title="Sika Spec Life Matrix", 
    page_icon="🏗️", 
    layout="wide"
)

# Lấy API Key an toàn từ Streamlit Secrets hoặc Environment Variable
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]

if not api_key:
    st.error("⚠️ **Chưa cấu hình GEMINI_API_KEY!** Vui lòng kiểm tra lại phần **Secrets** trong Settings của Streamlit Cloud.")
    st.stop()

# Cấu hình API Key cho SDK
genai.configure(api_key=api_key)

# System Prompt kiểm soát ảo giác (Temperature = 0.0)
STRICT_SYSTEM_PROMPT = """
Bạn là Chuyên gia Tư vấn Kỹ thuật Vật liệu Sika (Sika Specification Agent).
Nhiệm vụ của bạn là phân tích yêu cầu dự án và đề xuất giải pháp Sika DỰ TRÊN DỮ LIỆU ĐƯỢC CUNG CẤP TRONG CSDL (CONTEXT).

QUY TẮC BẮT BỘC (STRICT GROUNDING RULES):
1. CHỈ DÙNG THÔNG TIN CÓ TRONG CONTEXT: Tuyệt đối không tự bịa đặt tên sản phẩm, mã hiệu, tiêu chuẩn hoặc thông số kỹ thuật không xuất hiện trong phần CONTEXT dưới đây.
2. NẾU CSDL KHÔNG CÓ KẾT QUẢ: Nếu trong CONTEXT không tìm thấy sản phẩm Sika phù hợp với yêu cầu của người dùng, phải trả lời rõ: "Trong CSDL hiện tại của Sika chưa có tài liệu/sản phẩm phù hợp chính xác cho hạng mục này."
3. CHÍNH XÁC MÃ SẢN PHẨM: Mọi tên sản phẩm (như Sikafloor, Ucrete, Sikagard, SikaTop...) phải trích dẫn CHÍNH XÁC 100% từng ký tự theo tài liệu gốc.
4. KHÔNG TỰ SUY ĐOÁN: Không tự bịa đặt độ dầy hay định mức thi công nếu Context không đề cập.
"""

generation_config = {
    "temperature": 0.0,
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

# CHỦ ĐỘNG KHỞI TẠO MÔ HÌNH VỚI TÊN CHUẨN ĐÃ ĐƯỢC BẢO TRÌ
try:
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash",
        generation_config=generation_config
    )
except Exception:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config
    )

# ==========================================
# 2. CACHE CSDL FAISS VECTOR DB
# ==========================================
@st.cache_resource
def load_vector_store():
    if os.path.exists("index.faiss") and os.path.exists("index.pkl"):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            v_store = FAISS.load_local(
                folder_path=".", 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            return v_store
        except Exception as e:
            st.warning(f"⚠️ Không thể nạp FAISS Vector DB: {e}")
            return None
    return None

vector_store = load_vector_store()

# ==========================================
# 3. HÀM XỬ LÝ FILE ĐẦU VÀO
# ==========================================
def parse_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    content = ""
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            content = f"--- NỘI DUNG FILE BOQ ({uploaded_file.name}) ---\n" + df.to_string(index=False)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            content = f"--- NỘI DUNG FILE BOQ ({uploaded_file.name}) ---\n" + df.to_string(index=False)
        elif file_name.endswith('.pdf'):
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            content = f"--- NỘI DUNG FILE PDF ({uploaded_file.name}) ---\n"
            for i, page in enumerate(reader.pages):
                content += f"\n[Trang {i+1}]\n" + (page.extract_text() or "")
        elif file_name.endswith(('.txt', '.md')):
            content = uploaded_file.read().decode("utf-8")
    except Exception as e:
        content = f"[Lỗi đọc file: {str(e)}]"
    return content

# ==========================================
# 4. GIAO DIỆN CHÍNH (STREAMLIT UI)
# ==========================================
st.title("🏗️ SIKA SPEC LIFE MATRIX - AI SPECIFICATION AGENT")
st.caption("Hệ thống tự động thẩm định & Khớp giải pháp vật liệu Sika cho Dự án Công nghiệp")

if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📥 1. Đầu vào Yêu cầu Dự án")
    
    uploaded_file = st.file_uploader(
        "📁 Tải lên File Specs / BOQ (Excel, PDF, TXT)", 
        type=["xlsx", "xls", "csv", "pdf", "txt"]
    )
    
    manual_input = st.text_area(
        "✍️ Nhập/Bổ sung Yêu cầu Kỹ thuật Thủ công:",
        value=st.session_state.manual_text,
        placeholder="Ví dụ: Cần sàn Epoxy chống hóa chất cho nhà máy dược GMP, diện tích 2000m2, chịu tải trọng 5 tấn/m2...",
        height=140
    )
    
    st.markdown("---")
    st.markdown("💡 **Kịch bản Demo Nhanh:**")
    btn_c1, btn_c2, btn_c3 = st.columns(3)
    
    if btn_c1.button("🧪 Sàn Dược GMP"):
        st.session_state.manual_text = "Yêu cầu hệ thống sàn hoàn thiện cho nhà máy Dược phẩm GMP Châu Âu: Chịu rửa ma sát cao, kháng hóa chất vệ sinh, bề mặt phẳng mịn nhẵn bóng, độ dầy 3mm-4mm, tiêu chuẩn xả thải VOC thấp."
        st.rerun()
        
    if btn_c2.button("🏭 Bán dẫn ESD"):
        st.session_state.manual_text = "Dự án nhà máy đóng gói Bán dẫn SHTP: Yêu cầu hệ thống sàn ESD kiểm soát tĩnh điện (10^6 - 10^9 Ohms), chịu tải nhẹ đến trung bình, không phát sinh bụi sạch Class 100."
        st.rerun()
        
    if btn_c3.button("💧 Bể Nước Thải"):
        st.session_state.manual_text = "Hạng mục bể chứa nước thải công nghiệp pH 2-12: Yêu cầu vật liệu bọc phủ chống thấm và kháng mài mòn sinh học, thi công được trên bề mặt bê tông ẩm."
        st.rerun()

    submit_btn = st.button("⚡ Phân tích & Đề xuất Giải pháp", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 2. Kết quả Thẩm định Kỹ thuật")
    
    if submit_btn:
        file_content = parse_uploaded_file(uploaded_file) if uploaded_file else ""
        user_text = manual_input.strip() if manual_input else ""
        
        if not user_text and not file_content:
            st.warning("⚠️ Vui lòng tải lên file BOQ/Dự án hoặc nhập yêu cầu kỹ thuật thủ công.")
        else:
            with st.spinner("🤖 Sika AI Agent đang truy vấn Vector DB & lập luận..."):
                start_time = time.time()
                
                full_user_query = f"{user_text}\n\n[DỮ LIỆU FILE BOQ]:\n{file_content}" if file_content else user_text
                
                context = ""
                if vector_store is not None:
                    try:
                        docs = vector_store.similarity_search(full_user_query, k=5)
                        context = "\n---\n".join([doc.page_content for doc in docs])
                    except Exception as e:
                        context = f"[Lỗi truy vấn Vector DB: {str(e)}]"
                else:
                    context = "CSDL Vector DB đang ở trạng thái trực tiếp từ Input người dùng."

                full_prompt = f"""
{STRICT_SYSTEM_PROMPT}

==========================================
DỮ LIỆU CSDL SIKA ĐƯỢC TRÍCH XUẤT (CONTEXT):
{context}
==========================================

YÊU CẦU DỰ ÁN CỦA NGƯỜI DÙNG:
{full_user_query}

Hãy phân tích và đưa ra báo cáo thẩm định kỹ thuật chính xác theo đúng Context trên:
"""
                try:
                    response = model.generate_content(full_prompt)
                    elapsed_time = round(time.time() - start_time, 2)
                    
                    st.success(f"⚡ Hoàn thành xử lý AI thực tế trong **{elapsed_time} giây**")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối Gemini API: {str(e)}")
