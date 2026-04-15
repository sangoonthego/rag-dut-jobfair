from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import StorageContext, load_index_from_storage

from src.settings import init_settings
from src.index import STORAGE_DIR
from src.query import create_hybrid_query_engine 

from src.curriculum_handler import get_timeline_response
from src.job_handler import get_company_list_response

app = FastAPI(title="DUT JobFair RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    major: str = "Công nghệ thông tin" # Giá trị mặc định nếu Flutter quên gửi

query_engine = None

@app.on_event("startup")
async def startup_event():
    global query_engine
    print("1. Khởi tạo Settings (Ollama + BGE-M3)...")
    init_settings()
    
    print(f"2. Load dữ liệu từ {STORAGE_DIR}...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Lỗi Load Database. Hãy chạy file pipeline trước! Chi tiết: {e}")
        return
        
    print("3. Khởi tạo Hybrid Engine...")
    query_engine = create_hybrid_query_engine(index)
    print("API ĐÃ SẴN SÀNG NHẬN REQUEST TỪ FLUTTER!")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_msg = request.message.lower()
    print(f"ĐÃ NHẬN ĐƯỢC CÂU HỎI TỪ FLUTTER: '{request.message}' | NGÀNH: '{request.major}'")
    if any(kw in user_msg for kw in ["lộ trình", "môn học", "trên trường"]):
        print("-> [ROUTER] Đã điều hướng vào luồng TIMELINE JSON")
        response_data = get_timeline_response(request.major)
        return response_data
    
    elif any(kw in user_msg for kw in ["công ty", "tuyển dụng", "matching", "tuyển"]):
        print("-> [ROUTER] Đã điều hướng vào luồng COMPANY LIST WIDGET")
        # 💡 SỬA Ở ĐÂY: Dùng hàm get_company_list_response để lấy search_query xịn
        response_data = get_company_list_response(request.major)
        return response_data

    else:
        print("-> [ROUTER] Không khớp hardcode, kích hoạt LLAMA-INDEX RAG...")
        if not query_engine:
            raise HTTPException(status_code=500, detail="AI Engine chưa khởi tạo xong.")
        
        try:
            response = query_engine.query(request.message)
            
            citations = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    meta = node.node.metadata
                    # Đảm bảo handle lỗi nếu node.score bị None
                    score = round(node.score, 4) if node.score is not None else 0.0
                    citations.append({
                        "company": meta.get('company_name', 'N/A'),
                        "job": meta.get('job_title', 'N/A'),
                        "score": score
                    })
                
            # Lưu ý: Đổi key 'reply' thành 'content' và thêm 'type': 'text' 
            # để đồng bộ chuẩn format với các luồng ở trên cho Flutter dễ parse
            return {
                "type": "text",
                "content": str(response),
                "citations": citations
            }
        except Exception as e:
            print(f"[LỖI RAG] {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))