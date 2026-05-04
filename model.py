import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Load data with DuckDB ─────────────────────────────────────────────────
print("Loading data with DuckDB...")
con = duckdb.connect()
df = con.execute("""
    SELECT * FROM read_csv_auto('data/vessel_arrivals.csv')
""").df()
print(f"Loaded {len(df):,} records")

# ── Feature engineering ───────────────────────────────────────────────────
df['arrival_date'] = pd.to_datetime(df['arrival_date'])
df['week_of_year'] = df['arrival_date'].dt.isocalendar().week.astype(int)
df['is_peak_season'] = df['month'].isin([8, 9, 10]).astype(int)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_crisis_period'] = (
    (df['year'] == 2021) & (df['month'].isin([9,10,11,12]))).astype(int)

# Rolling 7-day average congestion per terminal
df = df.sort_values('arrival_date')
df['rolling_avg_turnaround'] = df.groupby('terminal')[
    'truck_turnaround_hrs'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())

# Encode categoricals
le_vessel = LabelEncoder()
le_terminal = LabelEncoder()
le_origin = LabelEncoder()
df['vessel_type_enc'] = le_vessel.fit_transform(df['vessel_type'])
df['terminal_enc'] = le_terminal.fit_transform(df['terminal'])
df['origin_enc'] = le_origin.fit_transform(df['origin_port'])

# ── Model 1: Predict truck turnaround time ────────────────────────────────
features = [
    'month', 'day_of_week', 'week_of_year', 'year',
    'teu_load', 'utilization_rate', 'dwell_time_days',
    'vessel_type_enc', 'terminal_enc', 'origin_enc',
    'is_peak_season', 'is_weekend', 'is_crisis_period',
    'rolling_avg_turnaround'
]

X = df[features]
y = df['truck_turnaround_hrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5,
    learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)

print(f"\n📊 TRUCK TURNAROUND PREDICTION RESULTS")
print("="*45)
print(f"Mean Absolute Error: {mae:.1f} hours")
print(f"R² Score:            {r2:.3f}")
print(f"Avg actual:          {y_test.mean():.1f} hrs")
print(f"Avg predicted:       {preds.mean():.1f} hrs")

# ── Feature importance ────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 TOP CONGESTION PREDICTORS:")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:28s} {bar} {row['importance']:.3f}")

# ── Generate 2025 forecast ────────────────────────────────────────────────
print("\nGenerating 2025 congestion forecast...")
future_dates = pd.date_range('2025-01-01', '2025-12-31', freq='D')
future_records = []

for date in future_dates:
    for terminal in df['terminal'].unique():
        terminal_avg = df[df['terminal'] == terminal][
            'rolling_avg_turnaround'].iloc[-7:].mean()
        future_records.append({
            'date': date,
            'terminal': terminal,
            'month': date.month,
            'day_of_week': date.dayofweek,
            'week_of_year': date.isocalendar()[1],
            'year': 2025,
            'teu_load': int(df['teu_load'].mean()),
            'utilization_rate': df['utilization_rate'].mean(),
            'dwell_time_days': df['dwell_time_days'].mean(),
            'vessel_type_enc': 1,
            'terminal_enc': le_terminal.transform([terminal])[0],
            'origin_enc': 5,
            'is_peak_season': int(date.month in [8,9,10]),
            'is_weekend': int(date.dayofweek >= 5),
            'is_crisis_period': 0,
            'rolling_avg_turnaround': terminal_avg,
        })

future_df = pd.DataFrame(future_records)
future_df['predicted_turnaround'] = model.predict(future_df[features])
future_df['congestion_level'] = pd.cut(
    future_df['predicted_turnaround'],
    bins=[0, 80, 100, 130, 999],
    labels=['Low', 'Moderate', 'High', 'Critical'])

# ── Print forecast summary ────────────────────────────────────────────────
print(f"\n📦 2025 CONGESTION FORECAST SUMMARY")
print("="*45)
monthly = future_df.groupby('month').agg(
    avg_turnaround=('predicted_turnaround','mean'),
    max_turnaround=('predicted_turnaround','max'),
).round(1)
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
for i, row in monthly.iterrows():
    bar = '█' * int(row['avg_turnaround'] / 5)
    print(f"  {month_names[i-1]}: {bar} {row['avg_turnaround']:.1f} hrs")

# ── Export for Tableau ────────────────────────────────────────────────────
# Daily summary
daily_summary = df.groupby(['arrival_date','terminal']).agg(
    vessel_calls=('vessel_id','count'),
    avg_turnaround=('truck_turnaround_hrs','mean'),
    avg_dwell=('dwell_time_days','mean'),
    total_teu=('teu_load','sum'),
    delay_count=('is_delayed','sum'),
).reset_index()
daily_summary['delay_rate'] = (
    daily_summary['delay_count'] / daily_summary['vessel_calls']).round(3)
daily_summary.to_csv('tableau_daily.csv', index=False)

# Monthly summary
monthly_summary = df.groupby(['year','month','terminal']).agg(
    vessel_calls=('vessel_id','count'),
    avg_turnaround=('truck_turnaround_hrs','mean'),
    avg_dwell=('dwell_time_days','mean'),
    total_teu=('teu_load','sum'),
    delay_rate=('is_delayed','mean'),
).reset_index()
monthly_summary['month_name'] = pd.to_datetime(
    monthly_summary['month'], format='%m').dt.strftime('%b')
monthly_summary.to_csv('tableau_monthly.csv', index=False)

# 2025 forecast
future_df[['date','terminal','month','predicted_turnaround',
           'congestion_level','is_peak_season']].to_csv(
    'tableau_forecast_2025.csv', index=False)

print("\n✅ Exported tableau_daily.csv")
print("✅ Exported tableau_monthly.csv")
print("✅ Exported tableau_forecast_2025.csv")
print("\n🎯 Model training complete!")
