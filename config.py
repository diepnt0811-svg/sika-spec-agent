import os
# 1. Cấu hình đường dẫn hệ thống Google Drive
DRIVE_ROOT = "/content/drive/MyDrive"

# Thư mục gốc dự án
PROJECT_DIR = os.path.join(DRIVE_ROOT, "My_AI_Agent_Project")

# Đường dẫn chuẩn tới thư mục chứa các file PDF Flooring
PDF_DATA_DIR = os.path.join(DRIVE_ROOT, "Spec 2.0 - Data Schema", "TMs", "Flooring")

# Thư mục lưu CSDL Vector vĩnh viễn trên Drive (nằm trong My_AI_Agent_Project/3. faiss_drive_db)
VECTOR_DB_DIR = os.path.join(PROJECT_DIR, "3. faiss_drive_db")

# 2. Cấu hình Model & Chunking
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GEMINI_API_KEY = "AQ.Ab8RN6ImI3gU7ClqeNy38oSkSOPYIDqO5cLphilr_n0OvYKtyQ"
