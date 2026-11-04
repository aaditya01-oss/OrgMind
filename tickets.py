import re
from contextlib import closing

from pii_detector import mask_pii
from search import get_db_connection

PRIORITIES = ("low", "medium", "high")
STAFF_ROLES = ("admin", "member")   # customers only ever see their own tickets


def ticket_label(ticket_id):
    return f"TKT-{ticket_id:04d}"


def create_ticket_row(org_id, user_id, issue, priority):
    """Stores a ticket for this person in this organization. Returns (ticket_id, priority)."""
    priority = (priority or "").lower()
    if priority not in PRIORITIES:
        priority = "medium"
    safe_issue = mask_pii(issue)[0][:1000]   # personal details are never stored in plain text

    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tickets (org_id, created_by, issue, priority) VALUES (%s, %s, %s, %s) RETURNING id",
            (org_id, user_id, safe_issue, priority),
        )
        ticket_id = cur.fetchone()[0]
        conn.commit()
    return ticket_id, priority


def get_ticket_row(org_id, user_id, role, ticket_id):
    """The ticket, only if it is in this organization AND (the person is staff OR created it).
    Returns None for 'does not exist' and 'not yours' alike, so nobody can probe for other tickets."""
    digits = re.sub(r"\D", "", str(ticket_id))
    if not digits:
        return None

    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, status, priority, issue FROM tickets
            WHERE id = %s AND org_id = %s AND (%s OR created_by = %s)
            """,
            (int(digits), org_id, role in STAFF_ROLES, user_id),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "status": row[1], "priority": row[2], "issue": row[3]}