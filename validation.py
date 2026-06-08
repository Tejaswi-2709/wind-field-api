import numpy as np
import pandas as pd


# =========================================================
# METRICS (FIXED COLUMN NAMES)
# =========================================================
def compute_metrics(df):

    if df is None or len(df) == 0:
        return {"mae": None, "rmse": None, "bias": None}

    # FIX: correct column names
    if "sar_speed" not in df.columns or "era5_speed" not in df.columns:
        raise Exception("Missing required columns: sar_speed or era5_speed")

    sar = df["sar_speed"].to_numpy()
    era = df["era5_speed"].to_numpy()

    mask = ~np.isnan(sar) & ~np.isnan(era)

    sar = sar[mask]
    era = era[mask]

    if len(sar) == 0:
        return {"mae": None, "rmse": None, "bias": None}

    error = sar - era

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    print("\n===== VALIDATION METRICS =====")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"BIAS: {bias:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias
    }


# =========================================================
# SAR ↔ ERA5 MATCHING (FIXED + SAFE)
# =========================================================
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd


def match_sar_era5(df, era5_data):

    if df is None or len(df) == 0:
        return pd.DataFrame()

    required_cols = ["lat", "lon", "sar_speed", "sar_dir"]

    for c in required_cols:
        if c not in df.columns:
            raise Exception(f"Missing column: {c}")

    if not all(k in era5_data for k in ["lat", "lon", "speed"]):
        raise Exception("Invalid ERA5 structure")

    # =====================================================
    # ERA5 ARRAYS
    # =====================================================
    lats = np.array(era5_data["lat"])
    lons = np.array(era5_data["lon"])
    speed = np.array(era5_data["speed"])

    if speed.ndim == 3:
        speed = speed[0]

    print("\n===== ERA5 GRID CHECK =====")
    print("ERA5 lat range:", lats.min(), lats.max())
    print("ERA5 lon range:", lons.min(), lons.max())

    print("\n===== SAR CHECK =====")
    print("SAR lat range:", df["lat"].min(), df["lat"].max())
    print("SAR lon range:", df["lon"].min(), df["lon"].max())

    # =====================================================
    # FIX 1: FORCE SORTING (CRITICAL)
    # =====================================================
    lat_sort_idx = np.argsort(lats)
    lon_sort_idx = np.argsort(lons)

    lats = lats[lat_sort_idx]
    lons = lons[lon_sort_idx]
    speed = speed[np.ix_(lat_sort_idx, lon_sort_idx)]

    # =====================================================
    # FIX 2: SAFE INTERPOLATOR (NO HARD FAIL DROP)
    # =====================================================
    interp = RegularGridInterpolator(
        (lats, lons),
        speed,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    matched_rows = []

    # =====================================================
    # MATCH LOOP
    # =====================================================
    for _, row in df.iterrows():

        lat = row["lat"]
        lon = row["lon"]

        if pd.isna(lat) or pd.isna(lon):
            continue

        era_val = interp((lat, lon))

        # FIX 3: fallback instead of dropping silently
        if np.isnan(era_val):

            # try nearest neighbor fallback (VERY IMPORTANT)
            era_val = interp((lat, lon), method="nearest")

        row_copy = row.copy()
        row_copy["era5_speed"] = float(era_val)

        matched_rows.append(row_copy)

    out = pd.DataFrame(matched_rows)

    print(f"✅ Matched rows: {len(out)}")

    return out