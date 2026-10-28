import json
import os
import re
from typing import Any, Dict, List, Optional

class EnterpriseKnowledgeRetriever:
  """Bộ truy xuất tri thức tổng quát: Kết hợp Lọc từ khóa chính xác (Lexical Filter)
  và CSDL Vector (Vector Search) cho mọi doanh nghiệp/ngành nghề.
  """

  def __init__(
      self,
      vector_db: Any,
      packaged_context_path: Optional[str] = None,
      llm_client: Optional[Any] = None,
      domain_context: Optional[str] = None,
  ):
    self.vector_db = vector_db
    self.packaged_context_path = packaged_context_path
    self.llm_client = llm_client
    self.domain_context = domain_context or 'Doanh nghiệp & Kỹ thuật'
    self.dynamic_context = self._load_dynamic_context()

  def _load_dynamic_context(self) -> Dict[str, Any]:
    if self.packaged_context_path and os.path.exists(
        self.packaged_context_path
    ):
      try:
        with open(self.packaged_context_path, 'r', encoding='utf-8') as f:
          return json.load(f)
      except Exception as e:
        print(f'⚠️ [Retriever Warning] Không thể nạp Dynamic Context: {e}')
    return {}

  def dynamic_query_expansion(self, user_query: str) -> List[str]:
    keywords = set()
    clean_query = user_query.strip().lower()

    extracted_codes = re.findall(
        r'\b[A-Za-z0-9]+(?:[-_/\.][A-Za-z0-9]+)*\b', user_query
    )
    for code in extracted_codes:
      code_lower = code.lower()
      if len(code_lower) >= 2:
        keywords.add(code_lower)

    if self.llm_client:
      try:
        prompt = (
            f"Bạn là chuyên gia trong lĩnh vực '{self.domain_context}'. "
            f"Với từ khóa/câu hỏi: '{user_query}', hãy liệt kê 3-5 từ khóa kỹ thuật, "
            f"tên tiêu chuẩn hoặc thuật ngữ đồng nghĩa liên quan nhất. "
            f"Chỉ trả về danh sách phân tách bằng dấu phẩy."
        )
        response = self.llm_client.invoke(prompt)
        content = getattr(response, 'content', str(response))
        generated_terms = [
            term.strip().lower() for term in content.split(',') if term.strip()
        ]
        keywords.update(generated_terms)
      except Exception:
        pass

    keywords.add(clean_query)
    return list(keywords)

  def search_static_kb(
      self, query: str, expanded_keywords: List[str], top_k: int = 5
  ) -> List[Dict[str, Any]]:
    clean_query = query.strip().lower()
    exact_matches = []
    exact_match_ids = set()

    if len(clean_query) <= 5 or clean_query.isalnum():
      docstore = getattr(self.vector_db, 'docstore', None)
      if docstore and hasattr(docstore, '_dict'):
        pattern = re.compile(
            r'\b' + re.escape(clean_query) + r'\b', re.IGNORECASE
        )

        for doc_id, doc in docstore._dict.items():
          file_name = doc.metadata.get('source_file', '').lower()
          content = doc.page_content.lower()

          if pattern.search(content) or pattern.search(file_name):
            is_tech_doc = any(
                tag in file_name for tag in ['tds', 'certificate', 'pds', 'spec']
            )
            score = 0.1000 if is_tech_doc else 0.3000

            exact_matches.append({
                'source_type': 'Static_KB_Document',
                'file_name': doc.metadata.get('source_file', 'N/A'),
                'page': doc.metadata.get('page', 'N/A'),
                'relevance_score': score,
                'content': doc.page_content,
            })
            exact_match_ids.add(doc.page_content[:100])
            if len(exact_matches) >= top_k:
              break

    search_prompt = (
        f"{query} {' '.join(expanded_keywords)}"
        if len(clean_query) > 4
        else query
    )
    try:
      docs_with_score = self.vector_db.similarity_search_with_score(
          search_prompt, k=top_k * 2
      )
    except Exception:
      docs_with_score = self.vector_db.similarity_search_with_score(
          query, k=top_k * 2
      )

    vector_results = []
    for doc, score in docs_with_score:
      if doc.page_content[:100] in exact_match_ids:
        continue

      vector_results.append({
          'source_type': 'Static_KB_Document',
          'file_name': doc.metadata.get('source_file', 'N/A'),
          'page': doc.metadata.get('page', 'N/A'),
          'relevance_score': round(float(score), 4),
          'content': doc.page_content,
      })

    combined_results = exact_matches + vector_results
    return combined_results[:top_k]

  def search_dynamic_context(
      self, keywords: List[str]
  ) -> List[Dict[str, Any]]:
    if not self.dynamic_context:
      return []

    matched_results = []
    data_nodes = self.dynamic_context.get(
        'projects', self.dynamic_context.get('data', {})
    )

    for group_name, files in data_nodes.items():
      if isinstance(files, dict):
        for file_key, file_data in files.items():
          data_str = json.dumps(file_data, ensure_ascii=False).lower()
          matches = [kw for kw in keywords if kw in data_str]
          if matches:
            matched_results.append({
                'source_type': 'Dynamic_Context_Data',
                'group_name': group_name,
                'item_key': file_key,
                'matched_keywords': matches,
                'data': file_data,
            })

    return matched_results

  def hybrid_retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
    expanded_keywords = self.dynamic_query_expansion(query)
    static_res = self.search_static_kb(query, expanded_keywords, top_k=top_k)
    dynamic_res = self.search_dynamic_context(expanded_keywords)

    return {
        'query_original': query,
        'extracted_keywords': expanded_keywords,
        'static_knowledge_results': static_res,
        'dynamic_context_results': dynamic_res,
    }


GLOBAL_RETRIEVER_INSTANCE: Optional[EnterpriseKnowledgeRetriever] = None

def init_retriever_tool(
    vector_db: Any,
    packaged_context_path: Optional[str] = None,
    llm_client: Optional[Any] = None,
    domain_context: Optional[str] = None,
):
  global GLOBAL_RETRIEVER_INSTANCE
  GLOBAL_RETRIEVER_INSTANCE = EnterpriseKnowledgeRetriever(
      vector_db=vector_db,
      packaged_context_path=packaged_context_path,
      llm_client=llm_client,
      domain_context=domain_context,
  )
  print('✅ Đã khởi tạo thành công Universal EnterpriseKnowledgeRetriever Tool!')
