import os
import sys

# Thiết lập đường dẫn gốc
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Cấu hình thư mục Cache cho HuggingFace để không tải lại model
cache_path = r"D:\huggingface_cache"
os.environ["HF_HOME"] = cache_path
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["TORCH_HOME"] = cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_path

import pymysql
import re
from bs4 import BeautifulSoup
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from src.settings import init_settings
from src.index import STORAGE_DIR
from src.pipelines.semantic_chunker import JobDescriptionChunker

def clean_html(raw_html: str) -> str:
    """
    Làm sạch các thẻ HTML cơ bản. Việc giải mã các ký tự phức tạp (unescaping) 
    và chuẩn hóa khoảng trắng sẽ do JobDescriptionChunker đảm nhận.
    """
    if not raw_html:
        return ""
        
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # Đổi thẻ <br> thành dấu xuống dòng thực sự
    for br in soup.find_all("br"):
        br.replace_with("\n")
    
    # Gắn dấu gạch ngang vào trước các thẻ <li> để giữ cấu trúc danh sách
    for li in soup.find_all("li"):
        li.insert(0, "- ")
        
    text = soup.get_text(separator="\n")
    
    # Bỏ các shortcode của WordPress (Đã được Regex cẩn thận hơn để tránh xóa nhầm text)
    # Tìm chính xác các shortcode cấu trúc như [vc_row] hoặc [/vc_column]
    text = re.sub(r'\[/?vc_[^\]]+\]', '', text) 
    
    return text

def run_pipeline():
    print("Khởi tạo cấu hình AI (Ollama + BGE-M3)...")
    init_settings()
    
    # Khởi tạo Chunker Ngữ nghĩa (Semantic Chunker) mới
    chunker = JobDescriptionChunker(target_size=400, overlap=0.15, min_chunk_size=50)
    
    print("Đang kết nối Database dut_jobfair...")
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='', 
        database='dut_jobfair',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    nodes = []
    
    try:
        with connection.cursor() as cursor:
            # Query lấy dữ liệu Job và Meta data
            sql = """
                SELECT 
                    p.ID,
                    p.post_title AS job_title,
                    p.post_content AS job_description,
                    MAX(CASE WHEN pm.meta_key = '_company_name' THEN pm.meta_value END) AS company_name,
                    MAX(CASE WHEN pm.meta_key = '_job_location' THEN pm.meta_value END) AS location,
                    MAX(CASE WHEN pm.meta_key = '_job_salary' THEN pm.meta_value END) AS salary
                FROM wp_posts p
                LEFT JOIN wp_postmeta pm ON p.ID = pm.post_id
                WHERE p.post_type = 'job_listing' 
                  AND p.post_status = 'publish'
                GROUP BY p.ID
            """
            cursor.execute(sql)
            records = cursor.fetchall()
            
            print(f"Tìm thấy {len(records)} công việc đang đăng! Bắt đầu phân mảnh (chunking) theo ngữ nghĩa...")
            
            for row in records:
                # 1. Làm sạch HTML thô
                raw_text = clean_html(row['job_description'])
                
                # Setup các biến Metadata
                company = row['company_name'] or "Chưa cập nhật công ty"
                title = row['job_title'] or "Chưa cập nhật vị trí"
                salary = row['salary'] or "Thỏa thuận"
                location = row['location'] or "Chưa cập nhật địa điểm"
                
                metadata = {
                    "company_name": company,
                    "job_title": title,
                    "salary": salary,
                    "location": location,
                    "job_id": row['ID']
                }
                
                # 2. Đính kèm thông tin Lương/Địa điểm vào đầu văn bản trước khi cắt
                header_text = f"Mức lương: {salary}\nĐịa điểm: {location}\n\n"
                full_text_to_chunk = header_text + raw_text
                
                # 3. Chạy qua Semantic Chunker
                chunk_texts = chunker.chunk_job_description(full_text_to_chunk, metadata)
                
                # 4. Tạo trực tiếp TextNodes (Bỏ qua khâu cắt mặc định của LlamaIndex)
                for chunk_text in chunk_texts:
                    node = TextNode(
                        text=chunk_text,
                        metadata=metadata,
                        # Ngăn LLM tự động nối chuỗi metadata rác vào đầu text 
                        # vì Chunker của mình đã tự tiêm Header [Company: X] vào rồi
                        excluded_llm_metadata_keys=["company_name", "job_title", "job_id", "salary", "location"],
                    )
                    nodes.append(node)
                
    finally:
        connection.close()
        
    print(f"Đã tạo ra {len(nodes)} chunks ngữ nghĩa. Đang nhúng (Embedding) vào Vector Database...")
    
    # Nạp trực tiếp list TextNodes vào Index
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(STORAGE_DIR)
    print(f"Hoàn tất! Dữ liệu đã được nạp thành công vào thư mục: {STORAGE_DIR}")

if __name__ == "__main__":
    run_pipeline()