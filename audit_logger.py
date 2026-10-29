import json
import logging
from contextlib import closing
from datetime import datetime, timezone

import psycopg2
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger("orgmind.audit")


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def log_event(
    event_type: str,
    user_id: str,
    org_id: str,
    details: dict,
    severity: str = "INFO"
):
    """Write one audit row. Never raises: an audit failure must not break the request,
    but it is logged loudly so you notice."""
    try:
        with closing(get_db_connection()) as conn:
            with conn:  # commits on success, rolls back on error
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO audit_logs
                        (event_type, user_id, org_id, details, severity, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        event_type,
                        user_id,
                        org_id,
                        json.dumps(details),
                        severity,
                        datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC, same as before
                    ))
    except Exception:
        logger.exception("AUDIT WRITE FAILED: %s user=%s org=%s", event_type, user_id, org_id)


def log_document_event(user_id: str, org_id: str, action: str, doc_name: str, status: str):
    log_event(
        event_type=f"DOCUMENT_{action.upper()}",
        user_id=user_id,
        org_id=org_id,
        details={"doc_name": doc_name, "status": status},
        severity="INFO" if status == "success" else "WARNING"
    )

def log_conversation(user_id: str, org_id: str, user_input: str, response_length: int):
    log_event(
        event_type="CONVERSATION",
        user_id=user_id,
        org_id=org_id,
        details={
            "input_length": len(user_input),
            "response_length": response_length,
            "input_preview": user_input[:50]
        },
        severity="INFO"
    )

def log_security_threat(user_id: str, org_id: str, threat_type: str, details: str):
    log_event(
        event_type="SECURITY_THREAT",
        user_id=user_id,
        org_id=org_id,
        details={
            "threat_type": threat_type,
            "details": details
        },
        severity="WARNING"
    )

def log_pii_detection(user_id: str, org_id: str, pii_types: list):
    log_event(
        event_type="PII_DETECTED",
        user_id=user_id,
        org_id=org_id,
        details={"pii_types": pii_types},
        severity="WARNING"
    )