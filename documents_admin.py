from contextlib import closing

from search import get_db_connection

VISIBILITIES = ("public", "registered", "internal", "department", "admin_only")


def list_documents_admin(org_id):
    """Every document of this organization with its label and how many text pieces it has."""
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.doc_name, d.visibility, d.department, d.uploaded_by, d.created_at,
                   (SELECT count(*) FROM document_chunks c
                    WHERE c.org_id = d.org_id AND c.doc_name = d.doc_name) AS chunks
            FROM documents d
            WHERE d.org_id = %s
            ORDER BY d.doc_name
            """,
            (org_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "doc_name": r[0],
            "visibility": r[1],
            "department": r[2],
            "uploaded_by": r[3],
            "created_at": r[4].isoformat() if r[4] else None,
            "chunks": r[5],
        }
        for r in rows
    ]


def set_visibility(org_id, doc_name, visibility, department):
    """Changes who can see one document. Returns how many documents were changed (0 or 1)."""
    visibility = (visibility or "").strip().lower()
    if visibility not in VISIBILITIES:
        raise ValueError("Invalid visibility choice.")
    department = (department or "").strip().lower() or None
    if visibility == "department" and not department:
        raise ValueError("Choose a department for a department-only document.")
    if visibility != "department":
        department = None

    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET visibility = %s, department = %s WHERE org_id = %s AND doc_name = %s",
            (visibility, department, org_id, doc_name),
        )
        changed = cur.rowcount
        conn.commit()
    return changed


def delete_document(org_id, doc_name):
    """Removes the document AND all its text pieces in one transaction, so the AI really forgets it.
    Returns how many documents were deleted (0 or 1)."""
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE org_id = %s AND doc_name = %s", (org_id, doc_name))
        cur.execute("DELETE FROM documents WHERE org_id = %s AND doc_name = %s", (org_id, doc_name))
        deleted = cur.rowcount
        conn.commit()
    return deleted