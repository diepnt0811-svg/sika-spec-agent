import os

def convert_table_to_markdown(table):
    if not table or len(table) < 2:
        return ""

    cleaned_table = []
    for row in table:
        cleaned_row = [
            str(cell).replace("\n", " ").strip() if cell is not None else "" 
            for cell in row
        ]
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)

    if len(cleaned_table) < 2:
        return ""

    header = cleaned_table[0]
    markdown_lines = []
    markdown_lines.append("| " + " | ".join(header) + " |")

    separator = "| " + " | ".join(["---"] * len(header)) + " |"
    markdown_lines.append(separator)

    for row in cleaned_table[1:]:
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))
        elif len(row) > len(header):
            row = row[:len(header)]
        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


def split_large_table_markdown(md_table, max_chars=2000):
    lines = md_table.split("\n")
    if len(lines) <= 2 or len(md_table) <= max_chars:
        return [md_table]

    header = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    chunks = []
    current_rows = []
    current_len = len(header) + len(separator)

    for row in data_rows:
        row_len = len(row) + 1
        if current_len + row_len > max_chars and current_rows:
            chunk_md = "\n".join([header, separator] + current_rows)
            chunks.append(chunk_md)
            current_rows = [row]
            current_len = len(header) + len(separator) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        chunk_md = "\n".join([header, separator] + current_rows)
        chunks.append(chunk_md)

    return chunks


def extract_pdf_with_tables(pdf_path):
    import pdfplumber
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    try:
        import config
        chunk_size = getattr(config, "CHUNK_SIZE", 1000)
        chunk_overlap = getattr(config, "CHUNK_OVERLAP", 200)
    except ImportError:
        chunk_size = 1000
        chunk_overlap = 200

    documents = []
    file_name = os.path.basename(pdf_path)
    folder_category = os.path.basename(os.path.dirname(pdf_path))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, 1):
                
                tables = page.extract_tables()
                if tables:
                    for t_idx, table in enumerate(tables, 1):
                        md_table = convert_table_to_markdown(table)
                        if md_table:
                            sub_tables = split_large_table_markdown(md_table, max_chars=2000)
                            
                            for sub_md in sub_tables:
                                table_content = f"[BẢNG THÔNG SỐ / KHÁNG HÓA CHẤT - TÀI LIỆU: {file_name} - TRANG {page_idx}]:\n\n{sub_md}"
                                
                                table_doc = Document(
                                    page_content=table_content,
                                    metadata={
                                        "source_file": file_name,
                                        "page_number": page_idx,
                                        "folder_category": folder_category,
                                        "is_table": True
                                    }
                                )
                                documents.append(table_doc)

                raw_text = page.extract_text()
                if raw_text and raw_text.strip():
                    raw_doc = Document(
                        page_content=raw_text.strip(),
                        metadata={
                            "source_file": file_name,
                            "page_number": page_idx,
                            "folder_category": folder_category,
                            "is_table": False
                        }
                    )
                    text_chunks = text_splitter.split_documents([raw_doc])
                    documents.extend(text_chunks)

    except Exception as e:
        print(f"⚠️ Lỗi khi đọc file {file_name}: {e}")

    return documents
