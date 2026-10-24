import json
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

def save_conversation(user_id, org_id, role, content):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversation_memory 
        (user_id, org_id, role, content, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (user_id, org_id, role, content, datetime.utcnow()))
    conn.commit()
    cur.close()
    conn.close()

def get_conversation_history(user_id, org_id, limit=10):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content, timestamp 
        FROM conversation_memory
        WHERE user_id = %s AND org_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """, (user_id, org_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

def save_user_profile(user_id, org_id, key, value):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_profiles (user_id, org_id, key, value, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, org_id, key) 
        DO UPDATE SET value = %s, updated_at = %s
    """, (user_id, org_id, key, value, datetime.utcnow(), value, datetime.utcnow()))
    conn.commit()
    cur.close()
    conn.close()

def get_user_profile(user_id, org_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT key, value FROM user_profiles
        WHERE user_id = %s AND org_id = %s
    """, (user_id, org_id))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {row[0]: row[1] for row in rows}