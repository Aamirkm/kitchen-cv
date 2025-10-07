import sqlite3
from datetime import datetime, timedelta, timezone

DB_FILE = "events.db"

def init_db():
    """Initializes the database and creates tables if they don't exist."""
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time DATETIME NOT NULL,
            end_time DATETIME,
            expected_thaals INTEGER,
            final_thaals_out INTEGER,
            final_thaals_in INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def create_session(session_id, expected_thaals=None):
    """Creates a new record for a service session using UTC time."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (session_id, start_time, expected_thaals) VALUES (?, ?, ?)",
                   (session_id, datetime.utcnow(), expected_thaals))
    conn.commit()
    conn.close()
    log_event("SERVICE_START", session_id)

def end_session(session_id):
    """Updates a session record with its end time in UTC."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET end_time = ? WHERE session_id = ?",
                   (datetime.utcnow(), session_id))
    conn.commit()
    conn.close()
    log_event("SERVICE_STOP", session_id)

def update_session_summary(session_id, out_count, in_count):
    """Updates the final thaal counts for a completed session."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET final_thaals_out = ?, final_thaals_in = ? WHERE session_id = ?",
                   (out_count, in_count, session_id))
    conn.commit()
    conn.close()

def log_event(event_type, session_id=None):
    """Logs an individual event to the events table using UTC time."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO events (timestamp, event_type, session_id) VALUES (?, ?, ?)",
                   (datetime.utcnow(), event_type, session_id))
    conn.commit()
    conn.close()

# --- Functions for Dashboard ---

def get_sessions_for_date(date_obj):
    """Finds all session data that occurred on a specific date, adjusted for local time."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    # THIS IS THE FIX: Use the 'localtime' modifier to convert UTC to local time before comparing the date.
    cursor.execute("SELECT session_id, start_time, end_time, expected_thaals, final_thaals_out, final_thaals_in FROM sessions WHERE DATE(start_time, 'localtime') = ?", (date_obj.strftime('%Y-%m-%d'),))
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_events_for_date(date_obj):
    """Gets all raw event data for a given date for CSV export, adjusted for local time."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    # THIS IS THE FIX: Also apply the 'localtime' modifier here.
    cursor.execute("SELECT * FROM events WHERE DATE(timestamp, 'localtime') = ? ORDER BY timestamp ASC", (date_obj.strftime('%Y-%m-%d'),))
    events = cursor.fetchall()
    conn.close()
    return events

def get_timeseries_data(session_ids):
    """Processes event data into a cumulative time-series for line charts."""
    if not session_ids:
        return [], [], []
        
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    placeholders = ','.join('?' for _ in session_ids)
    query = f"SELECT timestamp, event_type FROM events WHERE session_id IN ({placeholders}) AND (event_type = 'THAAL_OUT' OR event_type = 'THAAL_IN') ORDER BY timestamp ASC"
    
    cursor.execute(query, session_ids)
    events_utc = cursor.fetchall()
    conn.close()

    if not events_utc:
        return [], [], []

    # --- TIMEZONE FIX FOR CHARTS ---
    # Convert timestamps from UTC string to local datetime objects for processing
    local_events = []
    for ts_str, event_type in events_utc:
        # Create a timezone-aware datetime object for UTC
        utc_dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        # Convert to the system's local timezone
        local_dt = utc_dt.astimezone(None)
        local_events.append((local_dt, event_type))

    if not local_events:
        return [],[],[]
        
    start_time = local_events[0][0]
    end_time = local_events[-1][0]
    
    # Align start time to the beginning of a 5-minute interval for clean chart labels
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
    # Add a buffer to the end time to ensure the last interval is included
    while current_time <= end_time + timedelta(minutes=5):
        labels.append(current_time.strftime('%H:%M'))
        
        next_interval_time = current_time + timedelta(minutes=5)
        # Check against the timezone-aware local_events
        while event_index < len(local_events) and local_events[event_index][0] < next_interval_time:
            if local_events[event_index][1] == 'THAAL_OUT':
                count_out += 1
            elif local_events[event_index][1] == 'THAAL_IN':
                count_in += 1
            event_index += 1
        
        cumulative_out.append(count_out)
        cumulative_in.append(count_in)
        
        current_time = next_interval_time
        
    return labels, cumulative_out, cumulative_in

