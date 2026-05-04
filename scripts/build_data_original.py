import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import duckdb
import os

np.random.seed(42)

# ── Real Port of LA Annual TEU Data (from portoflosangeles.org) ──────────
annual_teu = {
    2010: 7831902, 2011: 7940511, 2012: 8077714, 2013: 7868582,
    2014: 8340066, 2015: 8160458, 2016: 8856783, 2017: 9343193,
    2018: 9458748, 2019: 9337632, 2020: 9213396, 2021: 10677610,
    2022: 9911159, 2023: 8629681, 2024: 10297352,
}

# ── Real monthly seasonality pattern (Port of LA historical pattern) ─────
# Imports peak Aug-Oct (holiday season prep), slow Jan-Feb
monthly_weights = {
    1: 0.072, 2: 0.071, 3: 0.081, 4: 0.082, 5: 0.085,
    6: 0.086, 7: 0.088, 8: 0.092, 9: 0.091, 10: 0.090,
    11: 0.083, 12: 0.079
}

# ── Vessel types at Port of LA ───────────────────────────────────────────
vessel_types = ['Ultra Large Container', 'Large Container',
                'Medium Container', 'Feeder Container', 'Ro-Ro']
vessel_weights = [0.25, 0.35, 0.25, 0.10, 0.05]

# Vessel capacity (TEUs) by type
vessel_capacity = {
    'Ultra Large Container': 18000,
    'Large Container':       12000,
    'Medium Container':       7500,
    'Feeder Container':       2500,
    'Ro-Ro':                  1000,
}

# ── Terminal operators at Port of LA ────────────────────────────────────
terminals = ['APM Terminals', 'TraPac', 'Yang Ming', 'Everport', 'China Shipping']

# ── Origin ports (major trade lanes to LA) ───────────────────────────────
origins = ['Shanghai', 'Shenzhen', 'Ningbo', 'Busan', 'Tokyo',
           'Singapore', 'Hong Kong', 'Taipei', 'Jakarta', 'Manila']

# ── Generate daily vessel arrivals 2019–2024 ────────────────────────────
print("Generating vessel arrival data...")
records = []
vessel_counter = 10000

for year in range(2019, 2025):
    annual = annual_teu[year]
    days_in_year = 366 if year % 4 == 0 else 365
    dates = pd.date_range(start=f'{year}-01-01',
                          end=f'{year}-12-31', freq='D')

    # COVID impact 2020 — reduced arrivals April-June
    covid_factor = 1.0

    for date in dates:
        month = date.month
        dow = date.dayofweek

        # Monthly weight
        m_weight = monthly_weights[month]

        # COVID slowdown
        if year == 2020 and month in [4, 5, 6]:
            covid_factor = 0.72
        elif year == 2020 and month in [7, 8]:
            covid_factor = 0.88
        elif year == 2021 and month in [9, 10, 11]:
            covid_factor = 0.78  # supply chain crisis backlog
        else:
            covid_factor = 1.0

        # Daily vessel count (Port of LA averages ~30 vessel calls/day)
        base_vessels = 28
        daily_vessels = int(np.random.poisson(
            base_vessels * m_weight * 12 * covid_factor))
        daily_vessels = max(5, min(daily_vessels, 55))

        for v in range(daily_vessels):
            vessel_type = np.random.choice(vessel_types, p=vessel_weights)
            capacity = vessel_capacity[vessel_type]

            # Utilization rate (85-98% for imports)
            utilization = np.random.uniform(0.82, 0.98)
            teu_load = int(capacity * utilization)

            # Dwell time (days vessel stays at berth)
            base_dwell = {'Ultra Large Container': 2.8,
                          'Large Container': 2.2,
                          'Medium Container': 1.8,
                          'Feeder Container': 1.2,
                          'Ro-Ro': 1.5}
            dwell_time = round(np.random.normal(
                base_dwell[vessel_type] * (1 + (1 - covid_factor)),
                0.4), 1)
            dwell_time = max(0.5, dwell_time)

            # Truck turnaround time (hours) — worsens with congestion
            congestion_factor = 1.0
            if year == 2021 and month in [9, 10, 11, 12]:
                congestion_factor = 2.8  # real 2021 port crisis
            elif year == 2020 and month in [4, 5]:
                congestion_factor = 0.6  # less traffic during COVID
            elif month in [8, 9, 10]:
                congestion_factor = 1.3  # peak season

            truck_turnaround = round(np.random.normal(
                90 * congestion_factor, 15), 1)
            truck_turnaround = max(30, truck_turnaround)

            # Delay flag (vessel arrives late)
            delay_prob = 0.15 * congestion_factor
            is_delayed = np.random.binomial(1, min(delay_prob, 0.95))
            delay_hours = round(np.random.exponential(8), 1) \
                if is_delayed else 0.0

            records.append({
                'vessel_id':        f'V{vessel_counter:06d}',
                'arrival_date':     date.strftime('%Y-%m-%d'),
                'year':             year,
                'month':            month,
                'day_of_week':      dow,
                'vessel_type':      vessel_type,
                'origin_port':      np.random.choice(origins),
                'terminal':         np.random.choice(terminals),
                'teu_load':         teu_load,
                'utilization_rate': round(utilization, 3),
                'dwell_time_days':  dwell_time,
                'truck_turnaround_hrs': truck_turnaround,
                'is_delayed':       bool(is_delayed),
                'delay_hours':      delay_hours,
                'congestion_factor':round(congestion_factor, 2),
            })
            vessel_counter += 1

df = pd.DataFrame(records)
print(f"Generated {len(df):,} vessel arrival records")
print(f"Date range: {df['arrival_date'].min()} to {df['arrival_date'].max()}")
print(f"Delay rate: {df['is_delayed'].mean():.1%}")
print(f"Avg truck turnaround: {df['truck_turnaround_hrs'].mean():.1f} hrs")
print(f"Avg dwell time: {df['dwell_time_days'].mean():.1f} days")

# ── Save to CSV (simulating S3 storage) ──────────────────────────────────
os.makedirs('data', exist_ok=True)
df.to_csv('data/vessel_arrivals.csv', index=False)
print(f"\n✅ Saved to data/vessel_arrivals.csv")
print(f"   File size: {os.path.getsize('data/vessel_arrivals.csv')/1e6:.1f} MB")

# ── Query with DuckDB (free Athena replacement) ───────────────────────────
print("\n🦆 Querying with DuckDB...")
con = duckdb.connect()

print("\nTop 5 most congested months:")
result = con.execute("""
    SELECT year, month,
           ROUND(AVG(truck_turnaround_hrs), 1) as avg_turnaround,
           ROUND(AVG(dwell_time_days), 2) as avg_dwell,
           ROUND(AVG(CAST(is_delayed AS INTEGER)) * 100, 1) as delay_pct,
           COUNT(*) as vessel_calls
    FROM read_csv_auto('data/vessel_arrivals.csv')
    GROUP BY year, month
    ORDER BY avg_turnaround DESC
    LIMIT 5
""").df()
print(result.to_string(index=False))

print("\nVessel calls by terminal:")
result2 = con.execute("""
    SELECT terminal,
           COUNT(*) as vessel_calls,
           ROUND(AVG(truck_turnaround_hrs), 1) as avg_turnaround,
           ROUND(AVG(teu_load), 0) as avg_teu
    FROM read_csv_auto('data/vessel_arrivals.csv')
    GROUP BY terminal
    ORDER BY vessel_calls DESC
""").df()
print(result2.to_string(index=False))

print("\nCOVID impact on delay rates:")
result3 = con.execute("""
    SELECT year,
           ROUND(AVG(CAST(is_delayed AS INTEGER)) * 100, 1) as delay_pct,
           ROUND(AVG(truck_turnaround_hrs), 1) as avg_turnaround,
           COUNT(*) as total_vessels
    FROM read_csv_auto('data/vessel_arrivals.csv')
    GROUP BY year
    ORDER BY year
""").df()
print(result3.to_string(index=False))

print("\n✅ DuckDB queries complete!")
