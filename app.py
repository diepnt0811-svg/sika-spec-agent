import os
import time
import pandas as pd
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Sika Spec Life Matrix", page_icon="🏗️", layout="wide")

# Cấu hình API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

STRICT_SYSTEM_PROMPT = """
Bạn là Chuyên gia Tư vấn Kỹ thuật Vật liệu Sika (Sika Specification Agent).
Nhiệm vụ của bạn là phân tích yêu cầu dự án và đề xuất giải pháp Sika DỰ TRÊN DỮ LIỆU ĐƯỢC CUNG CẤP TRONG CSDL (CONTEXT).

QUY TẮC BẮT BỘC (STRICT GROUNDING RULES):
1. CHỈ DÙNG THÔNG TIN CÓ TRONG CONTEXT: Tuyệt đối không tự bịa đặt tên sản phẩm, mã hiệu, tiêu chuẩn hoặc thông số kỹ thuật không xuất hiện trong phần CONTEXT dưới đây.
2. NẾU CSDL KHÔNG CÓ KẾT QUẢ: Nếu trong CONTEXT không tìm thấy sản phẩm Sika phù hợp với yêu cầu của người dùng, phải trả lời rõ: "Trong CSDL hiện tại của Sika chưa có tài liệu/sản phẩm phù hợp chính xác cho hạng mục này."
3. CHÍNH XÁC MÃ SẢN PHẨM: Mọi tên sản phẩm (như Sikafloor, Ucrete, Sikagard, SikaTop...) phải trích dẫn CHÍNH XÁC 100% từng ký tự theo tài liệu gốc.
4. KHÔNG TỰ SUY ĐOÁN: Không tự bịa đặt độ dầy hay định mức thi công nếu Context không đề cập.
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.0, "top_p": 0.95, "max_output_tokens": 2048}
)

@st.cache_resource
def load_vector_store():
    if api_key and os.path.exists("index.faiss") and os.path.exists("index.pkl"):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            return FAISS.load_local(folder_path=".", embeddings=embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Lỗi nạp FAISS Vector DB: {e}")
    return None

vector_store = load_vector_store()

# Giao diện
st.title("🏗️ SIKA SPEC LIFE MATRIX - AI SPECIFICATION AGENT")
st.caption("Hệ thống tự động thẩm định & Khớp giải pháp vật liệu Sika cho Dự án Công nghiệp")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📥 Đầu vào Yêu cầu Dự án")
    uploaded_file = st.file_uploader("Tải lên File Specs / BOQ (Excel, PDF, TXT)", type=["xlsx", "csv", "pdf", "txt"])
    manual_prompt = st.text_area("Nhập yêu cầu kỹ thuật thủ công:", placeholder="Ví dụ: Sàn Epoxy chống hóa chất nhà máy dược GMP 2000m2...", height=120)
    
    st.markdown("---")
    st.caption("💡 **Kịch bản Demo nhanh:**")
    if st.button("🧪 Sample 1: Sàn Sạch Dược phẩm GMP"):
        manual_prompt = "Yêu cầu hệ thống sàn hoàn thiện cho nhà máy Dược phẩm GMP Châu Âu: Chịu rửa ma sát cao, kháng hóa chất vệ sinh, bề mặt phẳng mịn nhẵn bóng, độ dầy 3mm-4mm."
    if st.button("🏭 Sample 2: Phòng Sạch Bán dẫn ESD"):
        manual_prompt = "Dự án nhà máy đóng gói Bán dẫn SHTP: Yêu cầu hệ thống sàn ESD kiểm soát tĩnh điện (10^6 - 10^9 Ohms), không phát sinh bụi sạch Class 100."

    submit_btn = st.button("⚡ Phân tích & Đề xuất Giải pháp", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Kết quả Thẩm định Kỹ thuật")
    if submit_btn:
        if not manual_prompt and not uploaded_file:
            st.warning("Vui lòng tải lên file hoặc nhập yêu cầu thủ công!")
        else:
            with st.spinner("🤖 Sika Agent đang trích xuất CSDL & Thẩm định..."):
                file_text = ""
                if uploaded_file:
                    if uploaded_file.name.endswith(".pdf"):
                        import pypdf
                        reader = pypdf.PdfReader(uploaded_file)
                        file_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    elif uploaded_file.name.endswith((".xlsx", ".csv")):
                        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                        file_text = df.to_string(index=False)
                    else:
                        file_text = uploaded_file.read().decode("utf-8")

                query = f"{manual_prompt}\n\n[DỮ LIỆU FILE]:\n{file_text}" if file_text else manual_prompt
                
                context = ""
                if vector_store:
                    docs = vector_store.similarity_search(query, k=5)
                    context = "\n---\n".join([doc.page_content for doc in docs])
                
                full_prompt = f"{STRICT_SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nYÊU CẦU DỰ ÁN:\n{query}"
                response = model.generate_content(full_prompt)
                
                st.markdown(response.text)
