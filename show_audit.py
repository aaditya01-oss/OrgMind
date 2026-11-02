import sys
import psycopg2
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
cur = conn.cursor()
cur.execute("SELECT * FROM audit_logs ORDER BY 1 DESC LIMIT %s", (limit,))
cols = [c[0] for c in cur.description]
print(cols)
for row in cur.fetchall():
    print(row)
cur.close()
conn.close()