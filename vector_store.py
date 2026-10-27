import os
import importlib
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config
import pdf_processor

def _process_single_pdf(pdf_path):
    import pdf_processor
    return pdf_path, pdf_processor.extract_pdf_with_tables(pdf_path)

class VectorDBManager:
    def __init__(self):
        importlib.reload(config)
        importlib.reload(pdf_processor)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Embedding Model đang chạy trên thiết bị: [{device.upper()}]")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )
        self.db_path = config.VECTOR_DB_DIR
        self.pdf_dir = config.PDF_DATA_DIR

    def load_db(self):
        # Kiểm tra sự tồn tại của file chỉ mục chính index.faiss
        index_file = os.path.join(self.db_path, "index.faiss")
        if os.path.exists(index_file):
            print(f"📖 Đang nạp CSDL Vector từ '{self.db_path}'...")
            return FAISS.load_local(
                self.db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print(f"⚠️ CSDL Vector chưa tồn tại hoặc chưa hoàn chỉnh tại '{self.db_path}'!")
            return None

    def get_processed_files(self, vector_store):
        processed_files = set()
        if vector_store and hasattr(vector_store, "docstore"):
            for doc_id, doc in vector_store.docstore._dict.items():
                if isinstance(doc, dict):
                    meta = doc.get("metadata", {})
                else:
                    meta = getattr(doc, "metadata", {})
                if "source_file" in meta:
                    processed_files.add(meta["source_file"])
        return processed_files

    def _create_vector_in_batches(self, chunks, batch_size=16):
        total_chunks = len(chunks)
        vector_store = None

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            print(f" ⚡ Batch [{i + 1}-{min(i + batch_size, total_chunks)}/{total_chunks}]...")
            
            if vector_store is None:
                vector_store = FAISS.from_documents(batch_chunks, self.embeddings)
            else:
                batch_db = FAISS.from_documents(batch_chunks, self.embeddings)
                vector_store.merge_from(batch_db)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return vector_store

    def build_from_scratch(self):
        pdf_files = []
        for root, _, files in os.walk(self.pdf_dir):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, f))

        print(f"🔍 Tìm thấy tổng cộng {len(pdf_files)} file PDF trong thư mục nguồn.")
        if not pdf_files:
            return None

        max_workers = min(os.cpu_count() or 2, 4)
        print(f"⚡ Đang bóc tách Bảng Biểu & Văn bản song song trên {max_workers} nhân CPU...")
        
        all_docs = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single_pdf, pdf_path) for pdf_path in pdf_files]
            for idx, future in enumerate(as_completed(futures), 1):
                pdf_path, docs = future.result()
                file_name = os.path.basename(pdf_path)
                print(f" [{idx}/{len(pdf_files)}] Hoàn thành trích xuất: {file_name}")
                if docs:
                    all_docs.extend(docs)

        chunks = all_docs

        if not chunks:
            print("⚠️ BỎ QUA: Không trích xuất được văn bản nào!")
            return self.load_db()

        print(f"\n📦 Tổng cộng tạo ra {len(chunks)} Chunks tối ưu.")
        print("⚡ Đang tạo Vector Embedding...")
        
        vector_store = self._create_vector_in_batches(chunks, batch_size=16)

        if vector_store:
            vector_store.save_local(self.db_path)
            print(f"\n✅ ĐÃ HOÀN THÀNH! CSDL Vector khởi tạo vĩnh viễn tại:\n '{self.db_path}'")

        return vector_store

    def auto_update_or_load(self):
        index_file = os.path.join(self.db_path, "index.faiss")
        if not os.path.exists(index_file):
            print("⚡ CSDL FAISS chưa tồn tại. Bắt đầu khởi tạo từ đầu...")
            return self.build_from_scratch()

        print("⚡ Đang nạp CSDL Vector từ Drive để kiểm tra cập nhật...")
        vector_store = self.load_db()

        if vector_store is None:
            return self.build_from_scratch()

        all_pdf_files = []
        for root, _, files in os.walk(self.pdf_dir):
            for f in files:
                if f.lower().endswith('.pdf'):
                    all_pdf_files.append(os.path.join(root, f))

        existing_files = self.get_processed_files(vector_store)
        new_pdf_files = [f for f in all_pdf_files if os.path.basename(f) not in existing_files]

        if not new_pdf_files:
            print("✅ KHÔNG CÓ FILE MỚI! CSDL Vector đã ở phiên bản mới nhất, bỏ qua bước build lại.")
            return vector_store

        print(f"\n🆕 Phát hiện {len(new_pdf_files)} file PDF MỚI. Bắt đầu nạp bổ sung...")

        max_workers = min(os.cpu_count() or 2, 4)
        new_docs = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single_pdf, pdf_path) for pdf_path in new_pdf_files]
            for idx, future in enumerate(as_completed(futures), 1):
                pdf_path, docs = future.result()
                file_name = os.path.basename(pdf_path)
                print(f" [{idx}/{len(new_pdf_files)}] Nạp bổ sung: {file_name}")
                if docs:
                    new_docs.extend(docs)

        chunks = new_docs

        if not chunks:
            print("⚠️ CẢNH BÁO: File mới không chứa dữ liệu đọc được.")
            return vector_store

        print(f"📦 Đã trích xuất {len(chunks)} Chunks. Đang tạo Vector...")
        new_db = self._create_vector_in_batches(chunks, batch_size=16)
        vector_store.merge_from(new_db)

        vector_store.save_local(self.db_path)
        print(f"\n🎉 HOÀN THÀNH CẬP NHẬT! Đã nạp thêm {len(new_pdf_files)} file mới vào CSDL.")
        return vector_store
