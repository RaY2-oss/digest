import io, zipfile, requests, pandas as pd
from datetime import datetime, timedelta, timezone

now = datetime.now(timezone.utc)
a = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0) - timedelta(minutes=45)
ts = a.strftime("%Y%m%d%H%M%S")
r = requests.get(f"http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip", timeout=60)
print("tick", ts, r.status_code, len(r.content))
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open(z.namelist()[0]), sep="\t", header=None,
                 usecols=[4, 7, 8, 9, 10], names=["url", "v1t", "v2t", "v1l", "v2l"],
                 on_bad_lines="skip", low_memory=False, dtype=str, encoding_errors="replace")
print("strok:", len(df))

shown = 0
for _, row in df.iterrows():
    v2l = str(row["v2l"] or "")
    ccs = {e.split("#")[2] for e in v2l.split(";")
           if len(e.split("#")) > 3 and len(e.split("#")[2]) == 2}
    if len(ccs) >= 3 and "EDUCATION" in str(row["v2t"] or ""):
        print("\nURL:", str(row["url"])[:110])
        print("\n--- V1Themes (col 7, ispolzuetsya seychas) ---")
        print(str(row["v1t"])[:350])
        print("\n--- V2EnhancedThemes (col 8) ---")
        print(str(row["v2t"])[:600])
        print("\n--- V1Locations (col 9, ispolzuetsya seychas) ---")
        print(str(row["v1l"])[:450])
        print("\n--- V2EnhancedLocations (col 10) ---")
        print(str(row["v2l"])[:700])
        shown += 1
        break

if not shown:
    print("ne nashel podhodyashchey stroki")
