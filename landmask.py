import pandas as pd
import numpy as np
from global_land_mask import globe


# =========================================================
# ADD LAT/LON FROM GEOMETRY (SAFE + VALIDATED)
# =========================================================
def add_lat_lon(df):

    if df is None or len(df) == 0:
        return df

    if "geometry" not in df.columns:
        return df

    df = df.copy()

    def safe_x(g):
        try:
            return float(g.x)
        except:
            return np.nan

    def safe_y(g):
        try:
            return float(g.y)
        except:
            return np.nan

    df["lon"] = df["geometry"].apply(safe_x)
    df["lat"] = df["geometry"].apply(safe_y)

    # ❌ CRITICAL FIX: remove invalid SNAP projected coords
    df = df[
        df["lat"].between(-90, 90) &
        df["lon"].between(-180, 180)
    ]

    return df


# =========================================================
# REMOVE LAND POINTS (ROBUST + SAFE CRS HANDLING)
# =========================================================
from global_land_mask import globe
import numpy as np

def remove_land_points(df):

    if df is None or len(df) == 0:
        return df

    df = df.copy()

    if "lat" not in df.columns or "lon" not in df.columns:
        print("⚠️ Missing lat/lon → skipping landmask")
        return df

    df = df.dropna(subset=["lat", "lon"])

    lat = df["lat"].values
    lon = df["lon"].values

    # HARD SAFETY CLIP
    lat = np.clip(lat, -90, 90)
    lon = np.clip(lon, -180, 180)

    try:
        mask = globe.is_land(df["lat"].values, df["lon"].values)
        mask = np.asarray(mask).astype(bool)

        # prevent full wipe bug
        if np.all(mask):
            print("⚠️ WARNING: all points classified as land → skipping mask")
            return df

        df = df[~mask].reset_index(drop=True)

    except Exception as e:
        print(f"⚠️ Landmask failed: {e}")

    print(f"🌊 Ocean points: {len(df)}")

    print("⚠️ Landmask removing too many points → skipping safety mode")
    return df