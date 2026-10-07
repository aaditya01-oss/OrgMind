import psycopg2
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

def semantic_search(query, org_id, top_k=3):
    query_embedding = get_embedding(query)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT content, doc_name,
        1 - (embedding <=> %s::vector) as similarity
        FROM document_chunks
        WHERE org_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, org_id, query_embedding, top_k))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return results

if __name__ == "__main__":
    query = input("Search: ")
    org_id = input("Org ID: ")
    results = semantic_search(query, org_id)
    
    for i, (content, doc_name, similarity) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Document: {doc_name}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Content: {content[:200]}")