import re
from contextlib import closing

from audit_logger import log_event
from pii_detector import mask_pii
from search import get_db_connection

PRIORITIES = ("low", "medium", "high")


def ticket_label(ticket_id):
    return f"TKT-{ticket_id:04d}"


def list_departments(org_id):
    """The departments that really exist in this organization (at least one person belongs to them)."""
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT department FROM users
            WHERE org_id = %s AND department IS NOT NULL AND department <> ''
            ORDER BY department
            """,
            (org_id,),
        )
        return [row[0] for row in cur.fetchall()]


def create_ticket_row(org_id, user_id, issue, priority, department=None):
    """Stores a ticket. The department the model suggests is only kept if it is a real department
    of this organization; otherwise the ticket stays unrouted for the admin.
    Returns (ticket_id, priority, department)."""
    priority = (priority or "").lower()
    if priority not in PRIORITIES:
        priority = "medium"

    dept = (department or "").strip().lower()
    if dept not in list_departments(org_id):
        dept = None

    safe_issue = mask_pii(issue)[0][:1000]   # personal details are never stored in plain text

    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tickets (org_id, created_by, issue, priority, department) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (org_id, user_id, safe_issue, priority, dept),
        )
        ticket_id = cur.fetchone()[0]
        conn.commit()

    # The ticket number, priority and department only. The issue text is never put in the audit log.
    log_event("TICKET_CREATED", user_id, org_id,
              {"ticket_id": ticket_id, "priority": priority, "department": dept})
    return ticket_id, priority, dept


def get_ticket_row(org_id, user_id, role, ticket_id, department=None):
    """The ticket, only if it is in this organization AND the person is an admin, created it,
    or is an employee of the department it was routed to.
    Returns None for 'does not exist' and 'not yours' alike, so nobody can probe for other tickets."""
    digits = re.sub(r"\D", "", str(ticket_id))
    if not digits:
        return None

    dept_staff = role == "member" and bool(department)
    with closing(get_db_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, status, priority, issue, department FROM tickets
            WHERE id = %s AND org_id = %s
              AND (%s OR created_by = %s OR (%s AND department = %s))
            """,
            (int(digits), org_id, role == "admin", user_id, dept_staff, department),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "status": row[1], "priority": row[2], "issue": row[3], "department": row[4]}