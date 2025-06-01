import sqlite3
import threading
import json
from datetime import datetime

class SharedMemory:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path=":memory:"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedMemory, cls).__new__(cls)
                cls._instance._init_db(db_path)
            return cls._instance

    def _init_db(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                sender TEXT,
                format TEXT,
                intent TEXT,
                urgency TEXT,
                summary TEXT,
                metadata TEXT,
                thread_id TEXT,
                timestamp TEXT
            )
        ''')
        self.conn.commit()

    # ---------- EMAIL logging ----------
    def log_email(self, source_id, email_result: dict):
        self.conn.execute(
            '''
            INSERT INTO memory_log (
                source_id, sender, format, intent, urgency, summary, metadata, thread_id, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                source_id,
                email_result.get("sender"),
                "EMAIL",
                email_result.get("intent"),
                email_result.get("urgency"),
                email_result.get("summary"),
                json.dumps(email_result),
                email_result.get("sender") or source_id,
                email_result.get("timestamp") or datetime.utcnow().isoformat()
            )
        )
        self.conn.commit()

    # ---------- JSON logging ----------
    def log_json(self, source_id: str, json_result: dict):
        metadata = json_result.get("metadata", {})
        data = json_result.get("data", {})
        errors = json_result.get("errors", [])
        status = json_result.get("status", "unknown")

        summary = (
            f"Validated: {list(data.keys())}"
            if status == "success"
            else f"Errors: {errors}"
        )

        self.conn.execute(
            '''
            INSERT INTO memory_log (
                source_id, sender, format, intent, urgency, summary, metadata, thread_id, timestamp
            )
            VALUES (?, NULL, ?, NULL, NULL, ?, ?, ?, ?)
            ''',
            (
                source_id,
                "JSON",
                summary,
                json.dumps(json_result),
                source_id,  # Thread ID = source ID for JSON files
                datetime.utcnow().isoformat()
            )
        )
        self.conn.commit()

    # ---------- Querying ----------
    def retrieve_thread(self, thread_id):
        cursor = self.conn.execute(
            'SELECT * FROM memory_log WHERE thread_id = ?', (thread_id,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def _row_to_dict(self, row):
        return {
            "id": row[0],
            "source_id": row[1],
            "sender": row[2],
            "format": row[3],
            "intent": row[4],
            "urgency": row[5],
            "summary": row[6],
            "metadata": json.loads(row[7]),
            "thread_id": row[8],
            "timestamp": row[9],
        }
