import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime, timezone
from fetch_ais import fetch_vessels
from predict   import score_vessels

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
LIVE_PATH    = os.path.join(DATA_DIR, "vessels_live.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "vessels_history.csv")

def run():
    print(f"\n🚢 Pipeline started at {datetime.now()}")
    os.makedirs(DATA_DIR, exist_ok=True)

    vessels = fetch_vessels(duration_seconds=90)
    if not vessels:
        print("⚠ No vessels captured — keeping existing data")
        return

    df = score_vessels(vessels)
    if df.empty:
        print("⚠ Scoring returned empty dataframe")
        return

    df.to_csv(LIVE_PATH, index=False)
    print(f"✅ Live snapshot: {len(df)} vessels → {LIVE_PATH}")

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, df], ignore_index=True)
        history["snapshot_time"] = pd.to_datetime(history["snapshot_time"], utc=True)
        cutoff  = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
        history = history[history["snapshot_time"] >= cutoff]
    else:
        history = df.copy()

    history.to_csv(HISTORY_PATH, index=False)
    print(f"✅ History updated: {len(history)} total records")
    print(f"\n📊 Current Port Status:")
    print(f"   Vessels tracked:          {len(df)}")
    print(f"   Delayed/anchored:         {df['is_delayed'].sum()}")
    print(f"   Avg predicted turnaround: {df['predicted_turnaround'].mean():.1f} hrs")
    print(f"   Congestion levels:\n{df['congestion_level'].value_counts().to_string()}")

if __name__ == "__main__":
    run()
