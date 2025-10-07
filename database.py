import sqlite3
from datetime import datetime

DB_FILE = "events.db"

def init_db():
    """Initializes the database and creates the events table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_event(event_type, session_id=None):
    """Logs an event to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO events (timestamp, event_type, session_id) VALUES (?, ?, ?)",
                   (datetime.now(), event_type, session_id))
    conn.commit()
    conn.close()

def get_last_session_events():
    """Finds the last full session and returns all its event data."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    # Find the session_id of the most recent 'SERVICE_STOP' event
    cursor.execute("SELECT session_id FROM events WHERE event_type = 'SERVICE_STOP' ORDER BY timestamp DESC LIMIT 1")
    last_session = cursor.fetchone()

    if not last_session:
        conn.close()
        return None
    
    last_session_id = last_session[0]

    # Get all events for that session
    cursor.execute("SELECT timestamp, event_type FROM events WHERE session_id = ? ORDER BY timestamp ASC", (last_session_id,))
    session_events = cursor.fetchall()
    conn.close()
    
    return last_session_id, session_events
