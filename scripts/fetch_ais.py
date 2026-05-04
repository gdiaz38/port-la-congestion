import asyncio, websockets, json, os, ssl, certifi
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

PORT_BOX      = [[33.68, -118.32], [33.78, -118.10]]
ANCHORAGE_BOX = [[33.65, -118.26], [33.72, -118.18]]

AIS_SHIP_TYPE_MAP = {
    range(70, 80): 'Large Container',
    range(80, 90): 'Ultra Large Container',
    range(20, 30): 'Ro-Ro',
    range(60, 70): 'Medium Container',
}

def classify_vessel(ship_type_code):
    for r, label in AIS_SHIP_TYPE_MAP.items():
        if ship_type_code in r:
            return label
    return 'Feeder Container'

def is_in_box(lat, lon, box):
    return box[0][0] <= lat <= box[1][0] and box[0][1] <= lon <= box[1][1]

async def collect(duration_seconds=90):
    key     = os.getenv("AIS_API_KEY")
    vessels = {}

    # Build SSL context explicitly using certifi CA bundle
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    print(f"🛳 Connecting to AIS stream (collecting for {duration_seconds}s)...")
    try:
        async with websockets.connect(
            "wss://stream.aisstream.io/v0/stream",
            ssl=ssl_ctx,
            ping_interval=20
        ) as ws:
            await ws.send(json.dumps({
                "APIKey":             key,
                "BoundingBoxes":      [PORT_BOX],
                "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
            }))

            deadline = asyncio.get_event_loop().time() + duration_seconds

            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw   = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg   = json.loads(raw)
                    mtype = msg.get("MessageType", "")

                    if mtype == "PositionReport":
                        p    = msg["Message"]["PositionReport"]
                        mmsi = str(msg.get("MetaData", {}).get("MMSI", ""))
                        lat  = p.get("Latitude",  0)
                        lon  = p.get("Longitude", 0)
                        sog  = p.get("Sog", 0)

                        if mmsi:
                            if mmsi not in vessels:
                                vessels[mmsi] = {
                                    "mmsi":        mmsi,
                                    "lat":         lat,
                                    "lon":         lon,
                                    "sog":         sog,
                                    "in_anchorage":is_in_box(lat, lon, ANCHORAGE_BOX),
                                    "first_seen":  datetime.now(timezone.utc).isoformat(),
                                    "ship_type":   0,
                                    "vessel_name": msg.get("MetaData",{}).get("ShipName","Unknown"),
                                }
                            else:
                                vessels[mmsi].update({
                                    "lat": lat, "lon": lon, "sog": sog,
                                    "in_anchorage": is_in_box(lat, lon, ANCHORAGE_BOX),
                                })

                    elif mtype == "ShipStaticData":
                        s    = msg["Message"]["ShipStaticData"]
                        mmsi = str(msg.get("MetaData", {}).get("MMSI", ""))
                        if mmsi in vessels:
                            vessels[mmsi]["ship_type"]   = s.get("Type", 0)
                            vessels[mmsi]["vessel_name"] = s.get("Name","").strip() or \
                                                           vessels[mmsi]["vessel_name"]

                except asyncio.TimeoutError:
                    continue

    except Exception as e:
        print(f"⚠ AIS connection error: {e}")

    now     = datetime.now(timezone.utc)
    results = []
    for mmsi, v in vessels.items():
        vessel_type = classify_vessel(v.get("ship_type", 0))
        is_delayed  = v["in_anchorage"] and v["sog"] < 1.0

        results.append({
            "mmsi":           mmsi,
            "vessel_name":    v["vessel_name"],
            "vessel_type":    vessel_type,
            "lat":            round(v["lat"], 5),
            "lon":            round(v["lon"], 5),
            "speed_knots":    v["sog"],
            "in_anchorage":   v["in_anchorage"],
            "is_delayed":     is_delayed,
            "snapshot_time":  now.isoformat(),
            "date":           now.strftime("%Y-%m-%d"),
            "month":          now.month,
            "day_of_week":    now.weekday(),
            "week_of_year":   now.isocalendar()[1],
            "year":           now.year,
        })

    print(f"✅ Captured {len(results)} vessels near LA/Long Beach")
    print(f"   Anchored/delayed: {sum(1 for r in results if r['is_delayed'])}")
    return results

def fetch_vessels(duration_seconds=90):
    return asyncio.run(collect(duration_seconds))

if __name__ == "__main__":
    vessels = fetch_vessels()
    for v in vessels[:5]:
        print(v)
