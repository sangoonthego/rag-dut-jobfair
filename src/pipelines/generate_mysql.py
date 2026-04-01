import os

import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

cache_path = r"D:\huggingface_cache"
os.environ["HF_HOME"] = cache_path
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["TORCH_HOME"] = cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_path

# --- BÂY GIỜ MỚI ĐƯỢC PHÉP GỌI CÁC THƯ VIỆN KHÁC ---
import pymysql
import re
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex
from src.settings import init_settings
from src.index import STORAGE_DIR

def clean_html(raw_html):
    if not raw_html:
        return ""
        
    soup = BeautifulSoup(raw_html, "html.parser")
    
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, "- ")
        
    text = soup.get_text(separator="\n")
    
    text = re.sub(r'\[.*?\]', '', text)
    
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    
    return text

def run_pipeline():
    print("Khởi tạo cấu hình AI (Ollama + BGE-M3)...")
    init_settings()
    
    print("Đang kết nối Database dut_jobfair...")
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='', 
        database='dut_jobfair',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    documents = []
    
    try:
        with connection.cursor() as cursor:
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
                WHERE p.post_type = 'job_listing' -- Thay đổi nếu theme dùng tên khác (vd: 'job')
                  AND p.post_status = 'publish'
                GROUP BY p.ID
            """
            cursor.execute(sql)
            records = cursor.fetchall()
            
            print(f"Tìm thấy {len(records)} công việc đang đăng! Bắt đầu làm sạch và đóng gói...")
            
            for row in records:
                clean_desc = clean_html(row['job_description'])
                
                company = row['company_name'] or "Chưa cập nhật công ty"
                salary = row['salary'] or "Thỏa thuận"
                location = row['location'] or "Chưa cập nhật địa điểm"
                
                content = f"--- THÔNG TIN TUYỂN DỤNG ---\n"
                content += f"Công ty: {company}\n"
                content += f"Vị trí: {row['job_title']}\n"
                content += f"Mức lương: {salary}\n"
                content += f"Địa điểm: {location}\n"
                content += f"\n--- MÔ TẢ CÔNG VIỆC & YÊU CẦU ---\n{clean_desc}\n"
                
                metadata = {
                    "company": company,
                    "title": row['job_title'],
                    "salary": salary,
                    "location": location,
                    "job_id": row['ID']
                }
                
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)
                
    finally:
        connection.close()
        
    print(f"Đang nhúng {len(documents)} tài liệu vào Vector Database. Quá trình này có thể mất vài phút tùy tốc độ máy...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(STORAGE_DIR)
    print(f"Hoàn tất! Dữ liệu đã được nạp thành công vào thư mục: {STORAGE_DIR}")

if __name__ == "__main__":
    run_pipeline()