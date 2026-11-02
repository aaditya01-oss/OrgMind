import re
from contextlib import closing

import psycopg2
from embeddings import get_query_embedding
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# MIN_SIMILARITY = 0.8   # multilingual-e5 scores sit high; 0.8 keeps all real answers and drops some junk
MIN_SIMILARITY = 0.7   # a low floor: the model itself says "not in the documents" when chunks don't answer
SOURCE_MARGIN = 0.025   # no longer used (sources are chosen by answer overlap); safe to delete later


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# ---------------------------------------------------------------------------
# WHO MAY SEE WHICH DOCUMENT
# ---------------------------------------------------------------------------
# Every search below is limited to documents this person may see. The rule lives HERE, in code and
# SQL, never in the prompt, so no wording of a question can talk its way around it.
#   admin       -> every document of the organization
#   everyone    -> 'public' and 'internal' documents
#   department  -> plus 'department' documents of their OWN department
# A document that has no row in the documents table is invisible (hidden by default).
# role/department default to the LOWEST access, so a caller that forgets them sees less, never more.
_VISIBLE = """
    doc_name IN (
        SELECT d.doc_name FROM documents d
        WHERE d.org_id = %s
          AND (
              %s
              OR d.visibility IN ('public', 'internal')
              OR (d.visibility = 'department' AND d.department = %s)
          )
    )
"""


def _viewer_params(org_id, role, department):
    return (org_id, role == "admin", department)


def semantic_search(query, org_id, top_k=5, doc_name=None, role="member", department=None):
    """Nearest chunks for the query, inside ONE organization, among documents this person may see.
    If doc_name is given (an EXACT stored name), only that document is searched."""
    query_embedding = get_query_embedding(query)
    viewer = _viewer_params(org_id, role, department)

    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        if doc_name is None:
            cur.execute(f"""
                SELECT content, doc_name,
                1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE org_id = %s AND {_VISIBLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, org_id, *viewer, query_embedding, top_k))
        else:
            cur.execute(f"""
                SELECT content, doc_name,
                1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE org_id = %s AND doc_name = %s AND {_VISIBLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, org_id, doc_name, *viewer, query_embedding, top_k))
        return cur.fetchall()


def list_document_names(org_id, role="member", department=None):
    """Names of the documents in this organization that this person may see."""
    viewer = _viewer_params(org_id, role, department)
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT doc_name FROM document_chunks WHERE org_id = %s AND {_VISIBLE} ORDER BY doc_name",
            (org_id, *viewer),
        )
        return [row[0] for row in cur.fetchall()]


def get_document_start(org_id, doc_name, max_chunks=3, role="member", department=None):
    """The first chunks of one document (what a person sees when skimming the top)."""
    viewer = _viewer_params(org_id, role, department)
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT content FROM document_chunks
            WHERE org_id = %s AND doc_name = %s AND {_VISIBLE}
            ORDER BY chunk_index
            LIMIT %s
            """,
            (org_id, doc_name, *viewer, max_chunks),
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


def get_document_sample(org_id, doc_name, n=8, role="member", department=None):
    """Chunks spread across the WHOLE document, in reading order.
    Returns (list_of_text, total_number_of_chunks)."""
    viewer = _viewer_params(org_id, role, department)
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM document_chunks WHERE org_id = %s AND doc_name = %s AND {_VISIBLE}",
            (org_id, doc_name, *viewer),
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