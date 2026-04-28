import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import os
from datetime import datetime

# ── Firebase Init ──────────────────────────────────────
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smartkitchensaftey-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# ── Fetch sensorHistory ────────────────────────────────
def fetch_history():
    print(">> Fetching sensorHistory from Firebase...")
    ref = db.reference('/sensorHistory')
    raw = ref.get()

    records = []
    if not raw:
        print("!! No history data found.")
        return pd.DataFrame()

    # Traverse: YYYY → MM → DD → HH → HH:MM → fields
    for year, months in raw.items():
        for month, days in months.items():
            for day, hours in days.items():
                for hour, entries in hours.items():
                    for time_key, vals in entries.items():
                        if not isinstance(vals, dict):
                            continue
                        records.append({
                            'datetime_str': f"{year}-{month}-{day} {vals.get('time', time_key)}",
                            'date':         vals.get('date', f"{year}-{month}-{day}"),
                            'time':         vals.get('time', time_key),
                            'mq2':          vals.get('mq2',         0),
                            'mq135':        vals.get('mq135',       0),
                            'temperature':  vals.get('temperature', 0),
                            'humidity':     vals.get('humidity',    0),
                            'status':       vals.get('status',      'Normal'),
                        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Parse datetime (handles AM/PM format)
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + 
                      df['time'].str.replace(r'\s*(AM|PM).*', '', regex=True),
                      errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sensor_history.csv', index=False)
    print(f">> Saved {len(df)} records → data/sensor_history.csv")
    return df

# ── Fetch alertsHistory ────────────────────────────────
def fetch_alerts():
    print(">> Fetching alertsHistory from Firebase...")
    ref = db.reference('/alertsHistory')
    raw = ref.get()

    records = []
    if not raw:
        print("!! No alert history found.")
        return pd.DataFrame()

    for year, months in raw.items():
        for month, days in months.items():
            for day, entries in days.items():
                for ts_key, vals in entries.items():
                    if not isinstance(vals, dict):
                        continue
                    records.append({
                        'date':        vals.get('date',        f"{year}-{month}-{day}"),
                        'time':        vals.get('time',        ts_key),
                        'type':        vals.get('type',        'UNKNOWN'),
                        'mq2':         vals.get('mq2',         0),
                        'mq135':       vals.get('mq135',       0),
                        'temperature': vals.get('temperature', 0),
                        'humidity':    vals.get('humidity',    0),
                    })

    df = pd.DataFrame(records)
    if not df.empty:
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/alert_history.csv', index=False)
        print(f">> Saved {len(df)} alert records → data/alert_history.csv")
    return df

# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    df_hist   = fetch_history()
    df_alerts = fetch_alerts()
    print("\n── Sensor History Sample ──")
    print(df_hist.tail(5).to_string(index=False))
    print("\n── Alert History Sample ──")
    print(df_alerts.tail(5).to_string(index=False))