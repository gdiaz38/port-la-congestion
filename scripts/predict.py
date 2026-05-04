import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle, pandas as pd, numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "delay_model.pkl")

def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def score_vessels(vessels: list) -> pd.DataFrame:
    artifact    = load_artifact()
    model       = artifact["model"]
    le_vessel   = artifact["le_vessel"]
    le_terminal = artifact["le_terminal"]
    le_origin   = artifact["le_origin"]
    features    = artifact["features"]
    origins     = artifact["origins"]

    df = pd.DataFrame(vessels)
    if df.empty:
        return df

    known_vessels = list(le_vessel.classes_)
    df["vessel_type_safe"] = df["vessel_type"].apply(
        lambda x: x if x in known_vessels else "Feeder Container")

    def assign_terminal(lon):
        if   lon < -118.28: return "APM Terminals"
        elif lon < -118.24: return "TraPac"
        elif lon < -118.20: return "Yang Ming"
        elif lon < -118.16: return "Everport"
        else:               return "China Shipping"

    df["terminal"]    = df["lon"].apply(assign_terminal)
    df["origin_port"] = np.random.choice(origins, size=len(df))

    capacity_map = {
        "Ultra Large Container": 18000,
        "Large Container":       12000,
        "Medium Container":       7500,
        "Feeder Container":       2500,
        "Ro-Ro":                  1000,
    }
    df["teu_load"]               = df["vessel_type_safe"].map(capacity_map) * 0.90
    df["utilization_rate"]       = 0.90
    df["dwell_time_days"]        = 2.0
    df["is_crisis_period"]       = 0
    df["rolling_avg_turnaround"] = 95.0
    df["is_peak_season"] = df["month"].isin([8,9,10]).astype(int)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)

    df["vessel_type_enc"] = le_vessel.transform(df["vessel_type_safe"])
    df["terminal_enc"]    = le_terminal.transform(df["terminal"])
    df["origin_enc"]      = le_origin.transform(df["origin_port"])

    df["predicted_turnaround"] = model.predict(df[features]).round(1)
    df["congestion_level"] = pd.cut(
        df["predicted_turnaround"],
        bins=[0, 80, 100, 130, 999],
        labels=["Low", "Moderate", "High", "Critical"]
    )
    return df

if __name__ == "__main__":
    from fetch_ais import fetch_vessels
    vessels = fetch_vessels(duration_seconds=30)
    result  = score_vessels(vessels)
    if not result.empty:
        print(result[["vessel_name","vessel_type","terminal",
                       "predicted_turnaround","congestion_level"]].to_string())
