import os
import sys
import time
import json
import re
import pandas as pd
import gradio as gr
from google import genai

# --- 1. CẤU HÌNH HỆ THỐNG & NHẬP MODULE NỘI BỘ ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import các module RAG & Engine nội bộ
try:
    from vector_store import VectorDBManager
    from knowledge_retriever_tool import EnterpriseKnowledgeRetriever, init_retriever_tool
    from agent_engine import SikaAIAgent
    from spec_matcher import SpecMatcher
except ImportError as e:
    print(f"⚠️ Cảnh báo Import Module: {e}")

# --- 2. KHỞI TẠO CSDL VECTOR & RETRIEVER ---
print("⏳ Đang khởi tạo CSDL FAISS Vector & Retriever...")
vdb_manager = VectorDBManager()
vector_db = vdb_manager.auto_update_or_load()

if vector_db is not None:
    init_retriever_tool(vector_db=vector_db)
    retriever = EnterpriseKnowledgeRetriever(vector_db=vector_db)
    agent = SikaAIAgent(vector_store=vector_db)
    matcher = SpecMatcher(retriever=retriever)
    print("✅ Đã khởi tạo Backend RAG Engine thành công!")
else:
    retriever = None
    print("❌ Không thể khởi tạo CSDL Vector FAISS!")

# Khởi tạo Google GenAI Client
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# --- 3. HÀM TRUY VẤN AI ENGINE (ĐÃ NÂNG CẤP GEMINI HIGH-VERSION & RETRY) ---
def ask_sika_agent_core(question: str) -> str:
    """
    Hàm xử lý truy vấn RAG chính kết hợp FAISS + Gemini API với cơ chế Retry & Fallback
    """
    if not retriever:
        return "❌ Hệ thống chưa sẵn sàng do chưa nạp được CSDL Vector FAISS."

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

    # 3. Danh sách Gemini Model ưu tiên phiên bản cao & Fallback
    model_list = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    response_text = None
    for model_name in model_list:
        for attempt in range(1, 4):
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
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg:
                    time.sleep(2)
                else:
                    break # Chuyển sang model tiếp theo
        if response_text:
            break

    if not response_text:
        return "⚠️ Tất cả các mô hình Gemini hiện đang bận hoặc không khả dụng. Vui lòng bấm gửi lại câu hỏi sau vài giây!"

    # Đính kèm danh sách tài liệu trích dẫn
    if static_docs:
        sources = list(set([f"{d.get('file_name')} (Trang {d.get('page')})" for d in static_docs]))
        response_text += "\n\n📌 **Tài liệu tham chiếu:**\n" + "\n".join([f"- {src}" for src in sources])

    return response_text

# --- 4. HÀM XỬ LÝ FILE UPLOAD & GRADIO CALLBACKS ---
def parse_uploaded_file(file_obj):
    if file_obj is None:
        return ""

    file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    ext = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if ext in ['.xlsx', '.xls', '.csv']:
            df = pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
            content = f"--- NỘI DUNG FILE BOQ/EXCEL ({os.path.basename(file_path)}) ---\n" + df.to_string(index=False)
        elif ext == '.pdf':
            import pypdf
            reader = pypdf.PdfReader(file_path)
            content = f"--- NỘI DUNG FILE PDF ({os.path.basename(file_path)}) ---\n"
            for i, page in enumerate(reader.pages):
                content += f"\n[Trang {i+1}]\n" + (page.extract_text() or "")
        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        content = f"[Lỗi đọc file: {str(e)}]"

    return content

def process_spec_request(file_obj, manual_prompt, history, progress=gr.Progress()):
    start_time = time.time()
    if history is None:
        history = []

    progress(0.1, desc="⏳ [1/3] Đang đọc & trích xuất dữ liệu File/BOQ...")
    file_content = parse_uploaded_file(file_obj) if file_obj else ""

    combined_query = ""
    if file_content:
        combined_query += f"Dữ liệu từ File dự án/BOQ:\n{file_content}\n\n"
    if manual_prompt.strip():
        combined_query += f"Yêu cầu bổ sung từ người dùng:\n{manual_prompt.strip()}"

    if not combined_query.strip():
        history.append({"role": "user", "content": "Thẩm định yêu cầu dự án"})
        history.append({"role": "assistant", "content": "⚠️ Vui lòng tải lên file BOQ/Dự án hoặc nhập yêu cầu kỹ thuật."})
        return history, "", "⚠️ Chưa nhập dữ liệu"

    progress(0.4, desc="🔍 [2/3] Đang truy vấn CSDL FAISS Vector & Đối chiếu Specs Sika...")
    user_msg = manual_prompt.strip() if manual_prompt.strip() else "Phân tích yêu cầu dự án và đề xuất giải pháp Sika phù hợp."

    progress(0.7, desc="🤖 [3/3] Gemini LLM đang tổng hợp & lập luận kết quả...")
    try:
        response = ask_sika_agent_core(combined_query)
        elapsed_time = round(time.time() - start_time, 2)
        status_info = f"⚡ Hoàn thành trong **{elapsed_time} giây**"

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})

        return history, response, status_info
    except Exception as e:
        elapsed_time = round(time.time() - start_time, 2)
        error_msg = f"❌ Lỗi xử lý từ hệ thống AI ({elapsed_time}s): {str(e)}"
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": error_msg})
        return history, error_msg, f"❌ Thất bại sau {elapsed_time}s"

def user_follow_up_chat(user_message, history, last_result, progress=gr.Progress()):
    if not user_message.strip():
        return "", history, ""

    start_time = time.time()
    if history is None:
        history = []

    progress(0.3, desc="🔍 Đang tìm thông tin bổ sung...")
    contextual_prompt = (
        f"Dựa trên kết quả thẩm định giải pháp Sika trước đó:\n\n{last_result}\n\n"
        f"Người dùng hỏi thêm: {user_message}"
    )

    progress(0.7, desc="🤖 Sika Agent đang suy luận câu trả lời...")

    try:
        bot_response = ask_sika_agent_core(contextual_prompt)
        elapsed_time = round(time.time() - start_time, 2)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})

        status_info = f"⚡ Phản hồi trong **{elapsed_time} giây**"
        return "", history, status_info
    except Exception as e:
        elapsed_time = round(time.time() - start_time, 2)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"❌ Lỗi: {str(e)}"})
        return "", history, f"❌ Lỗi sau {elapsed_time}s"

# --- 5. GIAO DIỆN GRADIO UI ---
with gr.Blocks() as demo:
    current_result_state = gr.State("")

    gr.Markdown(
        """
        # 🏗️ SIKA SPEC LIFE MATRIX - AI SPECIFICATION AGENT
        ### Hệ thống tự động thẩm định & Khớp giải pháp vật liệu Sika cho Dự án Công nghiệp
        """
    )

    with gr.Row():
        # CỘT TRÁI: DỮ LIỆU ĐẦU VÀO
        with gr.Column(scale=1):
            gr.Markdown("### 📥 1. Đầu vào Yêu cầu Dự án")

            file_input = gr.File(
                label="📁 Tải lên File Dự án / BOQ / Specs (Excel, PDF, TXT)",
                file_types=[".xlsx", ".xls", ".csv", ".pdf", ".txt"],
                type="filepath"
            )

            manual_input = gr.Textbox(
                label="✍️ Nhập/Bổ sung Yêu cầu Kỹ thuật Thủ công",
                placeholder="Ví dụ: Cần sàn Epoxy chống hóa chất cho nhà máy dược GMP, diện tích 2000m2...",
                lines=4
            )

            btn_submit = gr.Button("⚡ Phân tích & Đề xuất Giải pháp", variant="primary", size="lg")
            timer_output = gr.Markdown("⏱️ *Thời gian xử lý dự kiến: ~3 - 6 giây*")

            gr.Markdown("---")
            gr.Markdown("💡 **Kịch bản Demo Nhanh (Click để chọn):**")

            btn_sample_1 = gr.Button("🧪 Kịch bản 1: Sàn Sạch Dược phẩm GMP (Ucrete MF)")
            btn_sample_2 = gr.Button("🏭 Kịch bản 2: Phòng Sạch Bán dẫn SHTP (Chống tĩnh điện ESD)")
            btn_sample_3 = gr.Button("💧 Kịch bản 3: Chống thấm & Kháng hóa chất Bể xử lý nước thải")

        # CỘT PHẢI: KẾT QUẢ ĐẦU RA & Ô CHAT
        with gr.Column(scale=2):
            gr.Markdown("### 📊 2. Kết quả Phân tích & Khung Hỏi Đáp Trực tiếp")

            chatbot = gr.Chatbot(
                label="Lịch sử Phân tích & Thảo luận Kỹ thuật",
                height=450
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="💬 Hỏi đáp thêm về kết quả",
                    placeholder="So sánh chi phí, quy trình thi công, tiêu chuẩn ASTM...",
                    scale=4
                )
                btn_chat_send = gr.Button("Gửi 🚀", variant="secondary", scale=1)

    # NẮM BẮT SỰ KIỆN NÚT BẤM
    btn_submit.click(
        fn=process_spec_request,
        inputs=[file_input, manual_input, chatbot],
        outputs=[chatbot, current_result_state, timer_output]
    )

    btn_chat_send.click(
        fn=user_follow_up_chat,
        inputs=[chat_input, chatbot, current_result_state],
        outputs=[chat_input, chatbot, timer_output]
    )
    chat_input.submit(
        fn=user_follow_up_chat,
        inputs=[chat_input, chatbot, current_result_state],
        outputs=[chat_input, chatbot, timer_output]
    )

    # Kịch bản mẫu
    def load_sample_1():
        return None, "Yêu cầu hệ thống sàn hoàn thiện cho nhà máy Dược phẩm GMP Châu Âu: Chịu rửa ma sát cao, kháng hóa chất vệ sinh, bề mặt phẳng mịn nhẵn bóng, độ dầy 3mm-4mm, tiêu chuẩn xả thải VOC thấp."

    def load_sample_2():
        return None, "Dự án nhà máy đóng gói Bán dẫn SHTP: Yêu cầu hệ thống sàn ESD kiểm soát tĩnh điện (10^6 - 10^9 Ohms), chịu tải nhẹ đến trung bình, không phát sinh bụi sạch Class 100."

    def load_sample_3():
        return None, "Hạng mục bể chứa nước thải công nghiệp pH 2-12: Yêu cầu vật liệu bọc phủ chống thấm và kháng mài mòn sinh học, thi công được trên bề mặt bê tông ẩm."

    btn_sample_1.click(fn=load_sample_1, outputs=[file_input, manual_input])
    btn_sample_2.click(fn=load_sample_2, outputs=[file_input, manual_input])
    btn_sample_3.click(fn=load_sample_3, outputs=[file_input, manual_input])

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        debug=True,
        theme=gr.themes.Soft(primary_hue="yellow", secondary_hue="gray")
    )
