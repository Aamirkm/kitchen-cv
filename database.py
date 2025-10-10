import sqlite3
from datetime import datetime, timedelta
from collections import Counter

DB_FILE = "events.db"

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            expected_thaals INTEGER,
            final_thaals_out INTEGER,
            final_thaals_in INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def create_session(session_id, start_time, expected_thaals=None):
    """Creates a new record for a service session."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (session_id, start_time, expected_thaals) VALUES (?, ?, ?)",
                   (session_id, start_time.isoformat(), expected_thaals))
    conn.commit()
    conn.close()
    log_event("SERVICE_START", session_id)

def end_session(session_id, end_time):
    """Updates a session record with its end time."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET end_time = ? WHERE session_id = ?",
                   (end_time.isoformat(), session_id))
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
    """Logs an individual event to the events table."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO events (timestamp, event_type, session_id) VALUES (?, ?, ?)",
                   (datetime.now().isoformat(), event_type, session_id))
    conn.commit()
    conn.close()

# --- Functions for Dashboard ---

def get_sessions_for_date(date_obj):
    """Finds all session data that occurred on a specific date."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, start_time, end_time, expected_thaals, final_thaals_out, final_thaals_in FROM sessions WHERE DATE(start_time) = ?", (date_obj.strftime('%Y-%m-%d'),))
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_all_sessions():
    """Fetches all sessions from the database for a full export."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC")
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_events_for_date(date_obj):
    """Gets all raw event data for a given date for CSV export."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE DATE(timestamp) = ? ORDER BY timestamp ASC", (date_obj.strftime('%Y-%m-%d'),))
    events = cursor.fetchall()
    conn.close()
    return events

def get_duration_metrics(session_ids):
    """Calculates various duration metrics based on thaal out/in events."""
    if not session_ids:
        return 0, 0, 0

    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    placeholders = ','.join('?' for _ in session_ids)
    query = f"SELECT event_type, MIN(timestamp), MAX(timestamp) FROM events WHERE session_id IN ({placeholders}) AND event_type IN ('THAAL_OUT', 'THAAL_IN') GROUP BY event_type"
    
    cursor.execute(query, session_ids)
    results = cursor.fetchall()
    conn.close()

    timestamps = {
        'THAAL_OUT': {'min': None, 'max': None},
        'THAAL_IN': {'min': None, 'max': None}
    }

    for event_type, min_ts, max_ts in results:
        if min_ts:
            timestamps[event_type]['min'] = datetime.fromisoformat(min_ts)
        if max_ts:
            timestamps[event_type]['max'] = datetime.fromisoformat(max_ts)

    serving_duration = 0
    returning_duration = 0
    full_cycle_duration = 0

    # Duration of serving (first thaal out to last thaal out)
    if timestamps['THAAL_OUT']['min'] and timestamps['THAAL_OUT']['max']:
        serving_duration = round((timestamps['THAAL_OUT']['max'] - timestamps['THAAL_OUT']['min']).total_seconds() / 60)

    # Duration of returning (first thaal in to last thaal in)
    if timestamps['THAAL_IN']['min'] and timestamps['THAAL_IN']['max']:
        returning_duration = round((timestamps['THAAL_IN']['max'] - timestamps['THAAL_IN']['min']).total_seconds() / 60)
        
    # Duration of full cycle (first thaal out to last thaal in)
    if timestamps['THAAL_OUT']['min'] and timestamps['THAAL_IN']['max']:
        full_cycle_duration = round((timestamps['THAAL_IN']['max'] - timestamps['THAAL_OUT']['min']).total_seconds() / 60)

    return serving_duration, returning_duration, full_cycle_duration


def get_timeseries_data(session_ids):
    """Processes event data into a cumulative time-series for line charts."""
    if not session_ids:
        return [], [], []
        
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    placeholders = ','.join('?' for _ in session_ids)
    query = f"SELECT timestamp, event_type FROM events WHERE session_id IN ({placeholders}) AND (event_type = 'THAAL_OUT' OR event_type = 'THAAL_IN') ORDER BY timestamp ASC"
    
    cursor.execute(query, session_ids)
    events = cursor.fetchall()
    conn.close()

    if not events:
        return [], [], []

    local_events = [(datetime.fromisoformat(ts_str), event_type) for ts_str, event_type in events]
        
    start_time = local_events[0][0]
    end_time = local_events[-1][0]
    
    start_time -= timedelta(minutes=start_time.minute % 1,
                            seconds=start_time.second,
                            microseconds=start_time.microsecond)

    labels = []
    cumulative_out = []
    cumulative_in = []
    
    count_out = 0
    count_in = 0
    event_index = 0

    current_time = start_time
    while current_time <= end_time + timedelta(minutes=1):
        labels.append(current_time.strftime('%H:%M'))
        
        next_interval_time = current_time + timedelta(minutes=1)
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

def get_throughput_per_minute(session_ids):
    """Calculates the number of 'THAAL_OUT' events per minute for the given sessions."""
    if not session_ids:
        return {}
    
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    
    placeholders = ','.join('?' for _ in session_ids)
    query = f"SELECT timestamp FROM events WHERE session_id IN ({placeholders}) AND event_type = 'THAAL_OUT' ORDER BY timestamp ASC"
    
    cursor.execute(query, session_ids)
    events = cursor.fetchall()
    conn.close()
    
    if not events:
        return {}
        
    throughput = {}
    for (ts_str,) in events:
        event_time = datetime.fromisoformat(ts_str)
        minute_label = event_time.strftime('%H:%M')
        
        if minute_label not in throughput:
            throughput[minute_label] = 0
        throughput[minute_label] += 1
        
    return throughput

