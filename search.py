import re
from contextlib import closing

import psycopg2
from embeddings import get_embedding
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

MIN_SIMILARITY = 0.25   # chunks scoring below this are never shown to the model (chosen with docmind_eval.py)

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def semantic_search(query, org_id, top_k=5, doc_name=None):
    """Nearest chunks for the query, inside ONE organization.
    If doc_name is given (an EXACT stored name), only that document is searched."""
    query_embedding = get_embedding(query)

    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        if doc_name is None:
            cur.execute("""
                SELECT content, doc_name,
                1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE org_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, org_id, query_embedding, top_k))
        else:
            cur.execute("""
                SELECT content, doc_name,
                1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE org_id = %s AND doc_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, org_id, doc_name, query_embedding, top_k))
        return cur.fetchall()


def list_document_names(org_id):
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT doc_name FROM document_chunks WHERE org_id = %s ORDER BY doc_name",
            (org_id,),
        )
        return [row[0] for row in cur.fetchall()]


def get_document_start(org_id, doc_name, max_chunks=3):
    """The first chunks of one document (what a person sees when skimming the top)."""
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT content FROM document_chunks
            WHERE org_id = %s AND doc_name = %s
            ORDER BY chunk_index
            LIMIT %s
            """,
            (org_id, doc_name, max_chunks),
        )
        return [row[0] for row in cur.fetchall()]


def _squash(text):
    """'SRS Document.docx' -> 'srsdocumentdocx': lowercase letters and digits only
    (works for Devanagari too, because str.isalnum is Unicode-aware)."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_documents(names, hint):
    """Which stored names does a loose hint refer to? Pure Python, no SQL, so the
    model's text never goes anywhere near a query. Returns a list of exact names."""
    h = _squash(hint)
    if not h:
        return []
    exact = [n for n in names if _squash(n) == h]
    if exact:
        return exact
    return [n for n in names if h in _squash(n)]
def pick_indices(total, n):
    """n chunk positions spread evenly over 0..total-1, always including the first and last."""
    if total <= 0 or n <= 0:
        return []
    if total <= n:
        return list(range(total))
    if n == 1:
        return [0]
    return sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})


def get_document_sample(org_id, doc_name, n=8):
    """Chunks spread across the WHOLE document, in reading order.
    Returns (list_of_text, total_number_of_chunks)."""
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM document_chunks WHERE org_id = %s AND doc_name = %s",
            (org_id, doc_name),
        )
        total = cur.fetchone()[0]
        wanted = pick_indices(total, n)
        if not wanted:
            return [], 0
        cur.execute(
            """
            SELECT content FROM document_chunks
            WHERE org_id = %s AND doc_name = %s AND chunk_index = ANY(%s)
            ORDER BY chunk_index
            """,
            (org_id, doc_name, wanted),
        )
        return [row[0] for row in cur.fetchall()], total

if __name__ == "__main__":
    query = input("Search: ")
    org_id = input("Org ID: ")
    results = semantic_search(query, org_id)

    for i, (content, doc_name, similarity) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Document: {doc_name}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Content: {content[:200]}")