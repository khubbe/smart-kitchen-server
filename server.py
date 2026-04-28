from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import firebase_admin
from firebase_admin import credentials, db as fdb
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Firebase Init ─────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccount.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smartkitchensaftey-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

app = FastAPI(title="Smart Kitchen AI Server")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory cache ───────────────────────────
cache = {
    "summary":    {},
    "hourlyRisk": {},
    "anomalies":  {},
    "last_run":   None
}

# ── Safe recursive record extractor ──────────
def extract_records(node, depth=0, path_info=None):
    """
    Recursively walks Firebase tree.
    Collects any dict leaf that has mq2 + mq135.
    Safe against int/str/None at any level.
    """
    records = []
    if path_info is None:
        path_info = {}

    # If this node is a dict with sensor values → it's a record
    if isinstance(node, dict):
        if 'mq2' in node and 'mq135' in node:
            try:
                records.append({
                    'date':        str(node.get('date', path_info.get('date', '2026-01-01'))),
                    'time':        str(node.get('time', '--')),
                    'mq2':         float(node.get('mq2', 0)),
                    'mq135':       float(node.get('mq135', 0)),
                    'temperature': float(node.get('temperature', 0)),
                    'humidity':    float(node.get('humidity', 0)),
                    'status':      str(node.get('status', 'Normal')),
                })
            except (TypeError, ValueError):
                pass
            return records

        # Recurse into children
        for key, child in node.items():
            if not isinstance(child, (dict, list)):
                continue  # skip plain int/str values
            records.extend(extract_records(child, depth + 1, path_info))

    return records

# ── Main Analysis Function ────────────────────
def run_analysis():
    print(f"\n>> Auto analysis: {datetime.now().strftime('%H:%M:%S')}")
    try:
        raw = fdb.reference('/sensorHistory').get()
        if not raw:
            print("!! No history data yet")
            return

        if not isinstance(raw, dict):
            print("!! sensorHistory is not a dict — skipping")
            return

        # Extract all records safely
        records = extract_records(raw)

        if len(records) < 5:
            print(f"!! Not enough data yet ({len(records)} records)")
            return

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['mq2', 'mq135', 'temperature', 'humidity'])
        df = df[df['mq2'] > 0]  # filter zero-noise rows

        if len(df) < 5:
            print("!! Not enough valid rows after cleaning")
            return

        # ── Anomaly Detection (IsolationForest) ──
        features = ['mq2', 'mq135', 'temperature', 'humidity']
        X = StandardScaler().fit_transform(df[features].values)
        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        df['anomaly'] = model.fit_predict(X)
        anomalies = df[df['anomaly'] == -1]

        # ── Hourly Risk Pattern ───────────────────
        df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
        df['risk'] = (
            (df['mq2']         > 1500).astype(int) +
            (df['mq135']       > 2000).astype(int) +
            (df['temperature'] > 50  ).astype(int)
        )
        hourly = df.groupby('hour')['risk'].mean().reindex(range(24), fill_value=0)
        top_hour = int(hourly.idxmax())

        # ── Summary ───────────────────────────────
        alert_pct = round((df['status'] != 'Normal').mean() * 100, 1)
        summary = {
            'total_records':  int(len(df)),
            'anomaly_count':  int(len(anomalies)),
            'alert_rate_pct': alert_pct,
            'mq2_mean':       round(float(df['mq2'].mean()), 1),
            'mq2_max':        int(df['mq2'].max()),
            'mq135_mean':     round(float(df['mq135'].mean()), 1),
            'mq135_max':      int(df['mq135'].max()),
            'temp_mean':      round(float(df['temperature'].mean()), 1),
            'temp_max':       round(float(df['temperature'].max()), 1),
            'hum_mean':       round(float(df['humidity'].mean()), 1),
            'top_risk_hour':  top_hour,
            'last_run':       datetime.now().strftime('%I:%M %p  %d-%m-%Y')
        }

        hourly_dict = {
            str(i): round(float(hourly.get(i, 0)), 4)
            for i in range(24)
        }

        anom_dict = {}
        for i, (_, row) in enumerate(anomalies.tail(10).iterrows()):
            anom_dict[str(i)] = {
                'timestamp':   str(row.get('timestamp', '--'))[:19],
                'mq2':         int(row['mq2']),
                'mq135':       int(row['mq135']),
                'temperature': round(float(row['temperature']), 1),
                'humidity':    round(float(row['humidity']), 1),
                'status':      str(row['status'])
            }

        # ── Push to Firebase ──────────────────────
        ref = fdb.reference('/aiAnalysis')
        ref.child('summary').set(summary)
        ref.child('hourlyRisk').set(hourly_dict)
        ref.child('anomalies').set(anom_dict)

        # ── Update cache ──────────────────────────
        cache['summary']    = summary
        cache['hourlyRisk'] = hourly_dict
        cache['anomalies']  = anom_dict
        cache['last_run']   = summary['last_run']

        print(f">> Done | Records:{len(df)} Anomalies:{len(anomalies)} AlertRate:{alert_pct}%")

    except Exception as e:
        import traceback
        print(f"!! Analysis error: {e}")
        traceback.print_exc()

# ── Scheduler — every 5 min ───────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(run_analysis, 'interval', minutes=5)
scheduler.start()

# ── Run once on startup ───────────────────────
@app.on_event("startup")
async def startup():
    run_analysis()

# ── API Endpoints ─────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_run": cache["last_run"],
        "records": cache["summary"].get("total_records", 0),
        "anomalies": cache["summary"].get("anomaly_count", 0)
    }

@app.get("/api/summary")
def get_summary():
    return cache["summary"]

@app.get("/api/anomalies")
def get_anomalies():
    return cache["anomalies"]

@app.get("/api/hourly")
def get_hourly():
    return cache["hourlyRisk"]

@app.get("/api/history")
def get_history():
    try:
        raw = fdb.reference('/sensorData').get()
        return raw or {}
    except Exception as e:
        return {"error": str(e)}
