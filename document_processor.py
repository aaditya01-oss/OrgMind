import psycopg2
from pypdf import PdfReader
from embeddings import get_embedding
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def process_document(file_path, org_id, doc_name):
    extension = file_path.lower().split('.')[-1]
    
    if extension == 'pdf':
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
            
    elif extension == 'docx':
        from docx import Document
        doc = Document(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        
    elif extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    else:
        raise ValueError(f"Unsupported format: {extension}")
    
    chunks = chunk_text(full_text)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        cur.execute("""
            INSERT INTO document_chunks 
            (org_id, doc_name, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (org_id, doc_name, i, chunk, embedding))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Processed {len(chunks)} chunks from {doc_name}")

if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")
    org_id = input("Enter organization ID: ")
    doc_name = input("Enter document name: ")
    process_document(pdf_path, org_id, doc_name)