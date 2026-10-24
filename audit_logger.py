import psycopg2
from datetime import datetime
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

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
    conn = get_db_connection()
    cur = conn.cursor()
    
    import json
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
        datetime.utcnow()
    ))
    
    conn.commit()
    cur.close()
    conn.close()

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