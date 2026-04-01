import os
from typing import Any, Optional

cache_path = r"D:\huggingface_cache"
os.environ["HF_HOME"] = cache_path
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["TORCH_HOME"] = cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_path

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage

# Import các settings từ project
from src.settings import init_settings
from src.index import STORAGE_DIR

def create_query_engine(index: BaseIndex, **kwargs: Any) -> BaseQueryEngine:
    """
    Create a query engine for the given index.
    """
    # Lấy top 3 để tốc độ xử lý của model 1.5b nhanh hơn
    top_k = int(os.getenv("TOP_K", 3)) 
    if top_k != 0 and kwargs.get("filters") is None:
        kwargs["similarity_top_k"] = top_k

    kwargs["streaming"] = True
    return index.as_query_engine(**kwargs)

def get_query_engine_tool(
    index: BaseIndex,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> QueryEngineTool:
    if name is None:
        name = "query_index"
    if description is None:
        description = "Use this tool to retrieve information from a knowledge base."
    
    query_engine = create_query_engine(index, **kwargs)
    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=name,
        description=description,
    )

def chat_with_bot():
    print("1. Đang khởi tạo Ollama và Embedding...")
    init_settings() 
    
    print(f"2. Đang nạp Vector DB từ {STORAGE_DIR}...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Lỗi load DB: {e}")
        return

    print("3. Khởi tạo Engine (Streaming Mode)...")
    
    query_engine = create_query_engine(index)
    
    qa_prompt_str = (
        "Bạn là trợ lý ảo tuyển dụng DUT Job Fair. Dưới đây là thông tin công việc:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dựa vào thông tin trên, trả lời câu hỏi bằng TIẾNG VIỆT.\n"
        "Nếu không có thông tin, hãy nói 'Xin lỗi, tôi không thấy thông tin này'.\n"
        "Chỉ được sử dụng thông tin có trong ngữ cảnh. Nếu thông tin không có (như quyền lợi, địa điểm), tuyệt đối không tự ý thêm vào dựa trên kiến thức bên ngoài."
        "Câu hỏi: {query_str}\n"
        "Trả lời:"
    )
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": PromptTemplate(qa_prompt_str)
    })
    
    print("\n")
    print("CHATBOT DUT JOB FAIR (STREAMING) SẴN SÀNG!")
    
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        print("Bot: ", end="", flush=True)
        
        streaming_response = query_engine.query(user_input)
        
        for text in streaming_response.response_gen:
            print(text, end="", flush=True)
        
        print("\n") 

if __name__ == "__main__":
    chat_with_bot()