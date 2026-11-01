"""
Re-embed every stored chunk with the CURRENT model in embeddings.py.
Run once after changing the embedding model:   python reembed.py
The text (content) is not touched; only the vectors are replaced, one transaction.
"""
from contextlib import closing

import psycopg2
from embeddings import get_embeddings, MODEL_NAME
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def main():
    with closing(psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                                  user=DB_USER, password=DB_PASSWORD)) as conn:
        with conn:  # one transaction: all vectors change, or none do
            with conn.cursor() as cur:
                cur.execute("SELECT org_id, doc_name, chunk_index, content FROM document_chunks")
                rows = cur.fetchall()
                print(f"Re-embedding {len(rows)} chunks with {MODEL_NAME} ...")
                vectors = get_embeddings([r[3] for r in rows])
                cur.executemany(
                    "UPDATE document_chunks SET embedding = %s::vector "
                    "WHERE org_id = %s AND doc_name = %s AND chunk_index = %s",
                    [(vec, r[0], r[1], r[2]) for vec, r in zip(vectors, rows)],
                )
    print("Done.")


if __name__ == "__main__":
    main()