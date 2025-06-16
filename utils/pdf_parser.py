import fitz

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            'page_num': i + 1,
            'text': text.strip()
        })
    return pages