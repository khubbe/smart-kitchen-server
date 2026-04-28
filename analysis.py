import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import firebase_admin
from firebase_admin import credentials, db as fdb
import os, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
plt.style.use('dark_background')
COLORS = {'mq2':'#f97316','mq135':'#22d3ee','temperature':'#f43f5e','humidity':'#a78bfa'}

# ── Firebase Init ─────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccount.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smartkitchensaftey-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

# ── Load Data ─────────────────────────────────
def load():
    try:
        df = pd.read_csv('data/sensor_history.csv', parse_dates=['timestamp'])
        df = df.dropna(subset=['timestamp'])
        df[['mq2','mq135','temperature','humidity']] = \
            df[['mq2','mq135','temperature','humidity']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['mq2','mq135','temperature','humidity'])
        print(f">> Loaded {len(df)} records")
        return df
    except FileNotFoundError:
        print("!! Run fetch_data.py first!")
        return None

# ── 1. Sensor Trends ──────────────────────────
def plot_trends(df):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Sensor Trends Over Time', fontsize=15, color='white', y=1.01)
    sensors = [
        ('mq2','MQ2 (Smoke/Gas)',1500,'#f97316'),
        ('mq135','MQ135 (Air Quality)',2000,'#22d3ee'),
        ('temperature','Temperature (°C)',50,'#f43f5e'),
        ('humidity','Humidity (%)',95,'#a78bfa'),
    ]
    for ax, (col, label, thresh, color) in zip(axes, sensors):
        ax.plot(df['timestamp'], df[col], color=color, linewidth=1.2, alpha=0.85)
        ax.axhline(thresh, color='red', linestyle='--', alpha=0.6, linewidth=0.9)
        ax.set_ylabel(label, fontsize=8, color=color)
        ax.tick_params(colors='grey', labelsize=7)
        ax.fill_between(df['timestamp'], df[col], alpha=0.08, color=color)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=30, fontsize=7)
    plt.tight_layout()
    plt.savefig('outputs/01_sensor_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(">> Saved: outputs/01_sensor_trends.png")

# ── 2. Correlation Heatmap ────────────────────
def plot_correlation(df):
    cols = ['mq2','mq135','temperature','humidity']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.5, annot_kws={'size':11})
    ax.set_title('Sensor Correlation Heatmap', fontsize=13, color='white')
    plt.tight_layout()
    plt.savefig('outputs/02_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(">> Saved: outputs/02_correlation.png")

# ── 3. Anomaly Detection ──────────────────────
def detect_anomalies(df):
    features = ['mq2','mq135','temperature','humidity']
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    df = df.copy()
    df['anomaly'] = model.fit_predict(X_scaled)
    df['anomaly_score'] = model.decision_function(X_scaled)
    anomalies = df[df['anomaly'] == -1]
    print(f">> Anomalies detected: {len(anomalies)} / {len(df)}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['timestamp'], df['mq2'], color='#f97316', linewidth=1.0, alpha=0.7, label='MQ2')
    ax.scatter(anomalies['timestamp'], anomalies['mq2'],
               color='red', s=40, zorder=5, label='Anomaly', marker='x')
    ax.set_title('Anomaly Detection — MQ2 Sensor', fontsize=13, color='white')
    ax.legend(fontsize=9)
    ax.tick_params(colors='grey', labelsize=7)
    plt.tight_layout()
    plt.savefig('outputs/03_anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(">> Saved: outputs/03_anomaly_detection.png")

    anomalies[['timestamp','mq2','mq135','temperature','humidity','status']]\
        .to_csv('outputs/anomaly_report.csv', index=False)
    print(">> Saved: outputs/anomaly_report.csv")
    return df, anomalies

# ── 4. Hourly Risk ────────────────────────────
def plot_hourly_risk(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['risk'] = (
        (df['mq2']         > 1500).astype(int) +
        (df['mq135']       > 2000).astype(int) +
        (df['temperature'] > 50  ).astype(int)
    )
    hourly = df.groupby('hour')['risk'].mean().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hourly.index, hourly.values,
           color=['#ef4444' if v > 0.3 else '#f97316' if v > 0.1 else '#22d3ee'
                  for v in hourly.values])
    ax.set_title('Hourly Risk Pattern', fontsize=13, color='white')
    ax.set_xlabel('Hour of Day', color='grey')
    ax.set_ylabel('Risk Score', color='grey')
    ax.tick_params(colors='grey')
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.savefig('outputs/04_hourly_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(">> Saved: outputs/04_hourly_risk.png")
    return hourly

# ── 5. Status Distribution ────────────────────
def plot_status_dist(df):
    counts = df['status'].value_counts()
    colors_pie = ['#22d3ee','#f97316','#ef4444','#a78bfa','#f43f5e']
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(counts.values, labels=counts.index,
           colors=colors_pie[:len(counts)],
           autopct='%1.1f%%', startangle=140,
           textprops={'color':'white','fontsize':10})
    ax.set_title('System Status Distribution', fontsize=13, color='white')
    plt.tight_layout()
    plt.savefig('outputs/05_status_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(">> Saved: outputs/05_status_distribution.png")

# ── 6. Summary Stats ──────────────────────────
def get_summary(df):
    alert_pct = round((df['status'] != 'Normal').mean() * 100, 1)
    return {
        'total_records': int(len(df)),
        'alert_rate_pct': alert_pct,
        'mq2_mean':   round(float(df['mq2'].mean()), 1),
        'mq2_max':    int(df['mq2'].max()),
        'mq135_mean': round(float(df['mq135'].mean()), 1),
        'mq135_max':  int(df['mq135'].max()),
        'temp_mean':  round(float(df['temperature'].mean()), 1),
        'temp_max':   round(float(df['temperature'].max()), 1),
        'hum_mean':   round(float(df['humidity'].mean()), 1),
    }

# ── 7. Push to Firebase ───────────────────────
def push_to_firebase(df, anomalies, hourly):
    print("\n>> Pushing results to Firebase /aiAnalysis/ ...")
    ref = fdb.reference('/aiAnalysis')

    summary = get_summary(df)
    df_h = df.copy()
    df_h['hour'] = df_h['timestamp'].dt.hour
    df_h['risk'] = (
        (df_h['mq2']         > 1500).astype(int) +
        (df_h['mq135']       > 2000).astype(int) +
        (df_h['temperature'] > 50  ).astype(int)
    )
    top_hour = int(df_h.groupby('hour')['risk'].mean().idxmax())
    summary['top_risk_hour'] = top_hour
    summary['last_run'] = datetime.now().strftime('%I:%M %p  %d-%m-%Y')
    summary['anomaly_count'] = int(len(anomalies))

    # Hourly risk dict {"0": val, "1": val ...}
    hourly_dict = {str(i): round(float(hourly.get(i, 0)), 4) for i in range(24)}

    # Last 10 anomalies
    anom_list = {}
    top_anom = anomalies.tail(10).reset_index(drop=True)
    for i, row in top_anom.iterrows():
        anom_list[str(i)] = {
            'timestamp':   str(row['timestamp']),
            'mq2':         int(row['mq2']),
            'mq135':       int(row['mq135']),
            'temperature': round(float(row['temperature']), 1),
            'humidity':    round(float(row['humidity']), 1),
            'status':      str(row['status'])
        }

    ref.child('summary').set(summary)
    ref.child('hourlyRisk').set(hourly_dict)
    ref.child('anomalies').set(anom_list)
    print(">> Firebase /aiAnalysis/ updated!")
    print(f"   Records: {summary['total_records']} | Anomalies: {summary['anomaly_count']} | Alert Rate: {summary['alert_rate_pct']}%")

# ── Run All ───────────────────────────────────
if __name__ == "__main__":
    df = load()
    if df is not None and not df.empty:
        print("\n── Summary ──")
        s = get_summary(df)
        for k, v in s.items(): print(f"  {k}: {v}")

        plot_trends(df)
        plot_correlation(df)
        df_out, anomalies = detect_anomalies(df)
        hourly = plot_hourly_risk(df)
        plot_status_dist(df)
        push_to_firebase(df_out, anomalies, hourly)
        print("\n✅ All done! outputs/ folder + Firebase /aiAnalysis/ updated.")
    else:
        print("!! No data. Run fetch_data.py first.")