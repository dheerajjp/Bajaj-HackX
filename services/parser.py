from pdfminer.high_level import extract_text
from docx import Document

def parse_pdf(file_path):
    return extract_text(file_path)

def parse_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
