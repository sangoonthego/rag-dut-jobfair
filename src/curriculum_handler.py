import json
import os

def get_timeline_response(major: str):
    file_mapping = {
        "Công nghệ thông tin": "data/curriculum/cntt.json",
        "Quản lý dự án": "data/curriculum/qlda.json",
        "Tự động hóa": "data/curriculum/tdh.json"
    }
    
    file_path = file_mapping.get(major)
    
    if not file_path or not os.path.exists(file_path):
        return {
            "type": "text",
            "content": f"Hiện tại hệ thống chưa cập nhật chi tiết lộ trình cho ngành {major}. Bạn hỏi câu khác nhé!"
        }

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    timeline_steps = []
    lo_trinh = data.get("lo_trinh_theo_nam", {})
    
    for year_id, year_info in lo_trinh.items():
        mon_hoc_list = [mon["ten_mon"] for mon in year_info["cac_mon_cot_loi"]]
        desc_text = ", ".join(mon_hoc_list)
        
        year_number = year_id.split('_')[1] 
        
        timeline_steps.append({
            "title": f"Năm {year_number}: {year_info['muc_tieu_trong_tam']}", 
            "desc": desc_text,
            "project": year_info.get("do_an_thuc_te", "") 
        })

    return {
        "type": "widget",
        "action": "render_timeline",
        "major": major,
        "data": timeline_steps
    }