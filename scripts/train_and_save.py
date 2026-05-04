import pandas as pd
import numpy as np
import duckdb
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# Load existing synthetic data (last time we use it)
con = duckdb.connect()
df  = con.execute("SELECT * FROM read_csv_auto('data/vessel_arrivals.csv')").df()

df['arrival_date']  = pd.to_datetime(df['arrival_date'])
df['week_of_year']  = df['arrival_date'].dt.isocalendar().week.astype(int)
df['is_peak_season']   = df['month'].isin([8,9,10]).astype(int)
df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
df['is_crisis_period'] = ((df['year']==2021) & (df['month'].isin([9,10,11,12]))).astype(int)

df = df.sort_values('arrival_date')
df['rolling_avg_turnaround'] = df.groupby('terminal')['truck_turnaround_hrs'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())

le_vessel   = LabelEncoder()
le_terminal = LabelEncoder()
le_origin   = LabelEncoder()
df['vessel_type_enc'] = le_vessel.fit_transform(df['vessel_type'])
df['terminal_enc']    = le_terminal.fit_transform(df['terminal'])
df['origin_enc']      = le_origin.fit_transform(df['origin_port'])

features = [
    'month','day_of_week','week_of_year','year',
    'teu_load','utilization_rate','dwell_time_days',
    'vessel_type_enc','terminal_enc','origin_enc',
    'is_peak_season','is_weekend','is_crisis_period',
    'rolling_avg_turnaround'
]

X = df[features]
y = df['truck_turnaround_hrs']

model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X, y)

# Save model + encoders together
artifact = {
    'model':       model,
    'le_vessel':   le_vessel,
    'le_terminal': le_terminal,
    'le_origin':   le_origin,
    'features':    features,
    'terminals':   list(df['terminal'].unique()),
    'origins':     list(df['origin_port'].unique()),
}

os.makedirs('models', exist_ok=True)
with open('models/delay_model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("✅ Model saved to models/delay_model.pkl")
print(f"   Terminals: {artifact['terminals']}")
print(f"   Origins:   {artifact['origins']}")
