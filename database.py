import sqlite3
from datetime import datetime, timedelta

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

# --- NEW FUNCTIONS FOR DASHBOARD ---

def get_sessions_for_date(date_obj):
    """Finds all unique session IDs that occurred on a specific date."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM events WHERE DATE(timestamp) = ? AND session_id IS NOT NULL", (date_obj,))
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_session_events(session_id):
    """Gets all event data for a specific session_id."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, event_type FROM events WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    session_events = cursor.fetchall()
    conn.close()
    return session_id, session_events

def get_events_for_date(date_obj):
    """Gets all raw event data for a given date for CSV export."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE DATE(timestamp) = ? ORDER BY timestamp ASC", (date_obj,))
    events = cursor.fetchall()
    conn.close()
    return events

def get_timeseries_data(session_ids):
    """Processes event data into a cumulative time-series for line charts."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    # Create a placeholder string for the session IDs for the SQL query
    placeholders = ','.join('?' for _ in session_ids)
    query = f"SELECT timestamp, event_type FROM events WHERE session_id IN ({placeholders}) AND (event_type = 'THAAL_OUT' OR event_type = 'THAAL_IN') ORDER BY timestamp ASC"
    
    cursor.execute(query, session_ids)
    events = cursor.fetchall()
    conn.close()

    if not events:
        return [], [], []

    # --- Process data into 5-minute intervals ---
    start_time = datetime.fromisoformat(events[0][0])
    end_time = datetime.fromisoformat(events[-1][0])
    
    # Round start time down to the nearest 5 minutes
    start_time -= timedelta(minutes=start_time.minute % 5,
                            seconds=start_time.second,
                            microseconds=start_time.microsecond)

    labels = []
    cumulative_out = []
    cumulative_in = []
    
    count_out = 0
    count_in = 0
    event_index = 0

    current_time = start_time
    while current_time <= end_time:
        labels.append(current_time.strftime('%H:%M'))
        
        # Count events within this 5-minute window
        next_interval_time = current_time + timedelta(minutes=5)
        while event_index < len(events) and datetime.fromisoformat(events[event_index][0]) < next_interval_time:
            if events[event_index][1] == 'THAAL_OUT':
                count_out += 1
            elif events[event_index][1] == 'THAAL_IN':
                count_in += 1
            event_index += 1
        
        cumulative_out.append(count_out)
        cumulative_in.append(count_in)
        
        current_time = next_interval_time
        
    return labels, cumulative_out, cumulative_in

