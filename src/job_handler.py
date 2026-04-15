MAJOR_KEYWORDS = {
    "Công nghệ thông tin": "Developer Software Lập-trình Frontend Backend", 
    "Quản lý dự án": "Quản-lý-công-nghiệp Business-Analyst Supply-Chain",
    "Tự động hóa": "PLC SCADA Automation Robot"
}

def get_company_list_response(major: str):
    search_keyword = MAJOR_KEYWORDS.get(major, major)

    return {
        "type": "widget",
        "action": "render_company_list",
        "major": major,
        "search_query": search_keyword 
    }