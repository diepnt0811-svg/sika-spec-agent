import os
import sys
import importlib
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

# Thử import các module phụ trợ nếu có (không gây crash app nếu thiếu ở môi trường Cloud)
try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

try:
    import pdf_processor
    HAS_PDF_PROCESSOR = True
except ImportError:
    HAS_PDF_PROCESSOR = False

def _process_single_pdf(pdf_path):
    if HAS_PDF_PROCESSOR:
        import pdf_processor
        return pdf_path, pdf_processor.extract_pdf_with_tables(pdf_path)
    return pdf_path, []

class VectorDBManager:
    def __init__(self):
        # 1. Xác định đường dẫn gốc
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Đọc config nếu sẵn có
        if HAS_CONFIG:
            try:
                importlib.reload(config)
                self.model_name = getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
                self.db_path = getattr(config, "VECTOR_DB_DIR", self.base_dir)
                self.pdf_dir = getattr(config, "PDF_DATA_DIR", self.base_dir)
            except Exception:
                self.model_name = "BAAI/bge-m3"
                self.db_path = self.base_dir
                self.pdf_dir = self.base_dir
        else:
            self.model_name = "BAAI/bge-m3"
            self.db_path = self.base_dir
            self.pdf_dir = self.base_dir

        if HAS_PDF_PROCESSOR:
            try:
                importlib.reload(pdf_processor)
            except Exception:
                pass

        # 3. Cấu hình Device & Embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Embedding Model [{self.model_name}] đang chạy trên: [{device.upper()}]")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )

    def _resolve_db_path(self):
        """Tự động tìm kiếm vị trí chứa file index.faiss chính xác nhất"""
        possible_paths = [
            self.db_path,
            self.base_dir,
            os.getcwd(),
            os.path.join(self.base_dir, "vector_db"),
            os.path.join(os.getcwd(), "vector_db")
        ]
        
        for p in possible_paths:
            if p and os.path.exists(os.path.join(p, "index.faiss")):
                return p
        return self.db_path

    def load_db(self):
        """Nạp CSDL FAISS từ đĩa"""
        target_path = self._resolve_db_path()
        index_file = os.path.join(target_path, "index.faiss")
        
        if os.path.exists(index_file):
            print(f"📖 Đang nạp CSDL Vector từ '{target_path}'...")
            try:
                return FAISS.load_local(
                    target_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"❌ Lỗi nạp FAISS index tại {target_path}: {e}")
                return None
        else:
            print(f"⚠️ CSDL Vector chưa tồn tại hoặc không tìm thấy tại '{target_path}'!")
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
        if not HAS_PDF_PROCESSOR or not os.path.exists(self.pdf_dir):
            print("⚠️ Không tìm thấy module pdf_processor hoặc thư mục PDF gốc. Không thể build lại từ đầu.")
            return None

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
        # 1. Kiểm tra & nạp CSDL đã có
        resolved_path = self._resolve_db_path()
        index_file = os.path.join(resolved_path, "index.faiss")
        
        if os.path.exists(index_file):
            print(f"⚡ Đang nạp CSDL Vector FAISS từ '{resolved_path}'...")
            vector_store = self.load_db()
            if vector_store is not None:
                # Kiểm tra xem có môi trường để scan PDF mới không
                if HAS_PDF_PROCESSOR and os.path.exists(self.pdf_dir):
                    all_pdf_files = []
                    for root, _, files in os.walk(self.pdf_dir):
                        for f in files:
                            if f.lower().endswith('.pdf'):
                                all_pdf_files.append(os.path.join(root, f))

                    existing_files = self.get_processed_files(vector_store)
                    new_pdf_files = [f for f in all_pdf_files if os.path.basename(f) not in existing_files]

                    if new_pdf_files:
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

                        if new_docs:
                            print(f"📦 Đã trích xuất {len(new_docs)} Chunks. Đang tạo Vector...")
                            new_db = self._create_vector_in_batches(new_docs, batch_size=16)
                            vector_store.merge_from(new_db)
                            vector_store.save_local(resolved_path)
                            print(f"\n🎉 HOÀN THÀNH CẬP NHẬT! Đã nạp thêm {len(new_pdf_files)} file mới.")
                
                return vector_store

        # 2. Nếu không nạp được index.faiss có sẵn, chuyển sang build từ đầu (cho local/Colab)
        print("⚡ CSDL FAISS chưa tồn tại hoặc không thể nạp. Thử khởi tạo từ đầu...")
        return self.build_from_scratch()
