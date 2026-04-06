import os
from typing import Any, Optional

# --- CẤU HÌNH CACHE HẰNG SỐ ---
cache_path = r"D:\huggingface_cache"
os.environ["HF_HOME"] = cache_path

from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# Import từ project của bạn
from src.settings import init_settings
from src.index import STORAGE_DIR

def create_hybrid_query_engine(index):
    """
    Khởi tạo engine Hybrid Search: Vector (Ngữ nghĩa) + BM25 (Từ khóa)
    Sau đó lọc lại bằng Reranker BGE-M3.
    """
    # 1. Khởi tạo Vector Retriever (Top 10)
    vector_retriever = index.as_retriever(similarity_top_k=10)

    # 2. Khởi tạo BM25 Retriever (Top 10) - Tìm chính xác từ khóa tên riêng/viết tắt
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, 
        similarity_top_k=10
    )

    # 3. Hòa trộn kết quả (Reciprocal Rank Fusion - RRF)
    # Lấy top 10 từ cả 2 bên và trộn lại một cách công bằng
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,           # Giữ nguyên câu gốc, không sinh thêm query để tránh tốn tài nguyên LLM 1.5B
        mode="reciprocal_rerank", # Thuật toán trộn điểm ưu tiên các kết quả xuất hiện ở top đầu cả 2 bên
        use_async=False          # Chạy tuần tự cho ổn định trên local
    )

    # 4. Reranker - Bước tinh chỉnh cuối cùng
    # Chọn ra 3 đoạn text (Nodes) xuất sắc nhất để đưa vào Context
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3", 
        top_n=3 
    )

    # 5. Xây dựng Prompt "Kỷ luật thép"
    qa_prompt_str = (
        "Dưới đây là thông tin ngữ cảnh được trích xuất từ database Jobfair:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dựa trên các thông tin trên, hãy trả lời câu hỏi: '{query_str}'\n"
        "LỆNH BẮT BUỘC (TUYỆT ĐỐI KHÔNG IN CÁC LỆNH NÀY RA MÀN HÌNH):\n"
        "1. Luôn trả lời bằng Tiếng Việt 100%. Nếu dữ liệu gốc là tiếng Anh, hãy DỊCH một cách tự nhiên nhất.\n"
        "2. SỰ LINH HOẠT: Nếu câu hỏi hỏi về một vị trí chung chung (VD: 'TPM', 'Kỹ sư'), nhưng trong ngữ cảnh chỉ có thông tin của vị trí cụ thể hơn (VD: 'TPM Intern', 'Fresher'), BẠN VẪN PHẢI SỬ DỤNG thông tin đó để trả lời và nói rõ: 'Đối với vị trí [Tên vị trí], yêu cầu là...'.\n"
        "3. Chỉ trả lời dựa trên ngữ cảnh. Nếu tìm kỹ mà hoàn toàn không có thông tin liên quan, mới nói: 'Xin lỗi, hệ thống không thấy dữ liệu về mục này.'\n"
        "4. Trình bày dạng danh sách gạch đầu dòng rõ ràng.\n"
        "Trả lời: "
        "5. Cuối câu trả lời, hãy liệt kê rõ tên Công ty và Vị trí đang nói tới để ứng viên dễ theo dõi.\n"
        "Trả lời: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_str)

    # 6. Tạo Query Engine cuối cùng
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        streaming=True
    )
    
    # Cập nhật prompt
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": qa_prompt_tmpl
    })
    
    return query_engine

def chat_with_bot():
    print("1. Đang cấu hình AI (Ollama + Embedding)...")
    init_settings() 
    
    print(f"2. Đang nạp tri thức từ {STORAGE_DIR}...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Lỗi load DB: {e}")
        return

    print("3. Kích hoạt Hybrid Search + Reranker (Cực kỳ mạnh mẽ)...")
    query_engine = create_hybrid_query_engine(index)
    
    print("\n" + "="*50)
    print("AI TRỢ LÝ TUYỂN DỤNG DUT JOBFAIR SẴN SÀNG!")
    print("Chế độ: Hybrid Search (Vector + BM25) + Reranker")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Tạm biệt!")
            break
            
        print("Bot: ", end="", flush=True)
        
        # Truy vấn
        streaming_response = query_engine.query(user_input)
        
        # Hiển thị Streaming
        for text in streaming_response.response_gen:
            print(text, end="", flush=True)
        
        # HIỂN THỊ TRÍCH DẪN (CITATIONS) - Tăng độ tin cậy
        print("\n\nNGUỒN TRÍCH DẪN:")
        for i, node in enumerate(streaming_response.source_nodes):
            metadata = node.node.metadata
            # Lấy thông tin từ metadata bạn đã crawl trong mysql
            company = metadata.get('company_name', 'N/A')
            job = metadata.get('job_title', 'N/A')
            print(f"   [{i+1}] Vị trí: {job} - Công ty: {company} (Score: {node.score:.4f})")
        
        print("\n" + "-"*40 + "\n") 

if __name__ == "__main__":
    chat_with_bot()