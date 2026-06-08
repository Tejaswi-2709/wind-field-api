import xarray as xr
import numpy as np
from dotenv import load_dotenv

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()
import os
import cdsapi
from dotenv import load_dotenv

load_dotenv()


def get_era5_file(date, bbox):
    """
    Downloads ERA5 wind data for a given SAR-derived bbox.

    Parameters:
        date  -> "YYYY-MM-DD"
        bbox  -> [min_lon, min_lat, max_lon, max_lat]

    Returns:
        out_path (netcdf file)
    """

    # =====================================================
    # DIRECTORIES
    # =====================================================
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)

    # =====================================================
    # FILE NAME (DETERMINISTIC CACHE)
    # =====================================================
    date_str = date.replace("-", "")
    min_lon, min_lat, max_lon, max_lat = bbox

    filename = (
        f"era5_{date_str}_"
        f"{min_lon:.2f}_{min_lat:.2f}_"
        f"{max_lon:.2f}_{max_lat:.2f}.nc"
    )

    out_path = os.path.join(data_dir, filename)

    # =====================================================
    # CACHE CHECK (VERY IMPORTANT)
    # =====================================================
    if os.path.exists(out_path):
        print(f"✅ ERA5 file already exists: {filename}")
        return out_path

    # =====================================================
    # VALIDATE BBOX
    # =====================================================
    if min_lon >= max_lon or min_lat >= max_lat:
        raise Exception("❌ Invalid bbox for ERA5 download")

    print("\n===== ERA5 DOWNLOAD REGION =====")
    print(
        f"min_lon={min_lon:.3f}, min_lat={min_lat:.3f}, "
        f"max_lon={max_lon:.3f}, max_lat={max_lat:.3f}"
    )

    # =====================================================
    # CDS FORMAT: [north, west, south, east]
    # =====================================================
    area = [
        max_lat,
        min_lon,
        min_lat,
        max_lon
    ]

    # =====================================================
    # AUTH
    # =====================================================
    cds_url = os.getenv("CDS_URL")
    cds_key = os.getenv("CDS_KEY")

    if not cds_url or not cds_key:
        raise Exception("❌ Missing CDS_URL or CDS_KEY in .env")

    client = cdsapi.Client(url=cds_url, key=cds_key)

    # =====================================================
    # DOWNLOAD REQUEST
    # =====================================================
    print("📥 Downloading ERA5 (bbox-limited)...")

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": date[0:4],
            "month": date[5:7],
            "day": date[8:10],
            "time": [
                "00:00",
                "06:00",
                "12:00",
                "18:00"
            ],
            "format": "netcdf",
            "area": area
        },
        out_path
    )

    print(f"✅ ERA5 downloaded: {filename}")

    return out_path


# =====================================================
# DOWNLOAD ERA5 (GLOBAL PER DATE - FIXED)
# =====================================================



# =====================================================
# LOAD + FIX ERA5 + SUBSET TO SAR BBOX (CRITICAL FIX)
# =====================================================
def load_era5_wind_new(era5_path, bbox):
    import inspect
    print("ERA5 FUNCTION FROM:", inspect.getfile(inspect.currentframe()))
    """
    Loads ERA5 and correctly aligns it to SAR bbox.
    """

    print(f"\n📥 Loading ERA5: {era5_path}")

    min_lon, min_lat, max_lon, max_lat = bbox

    ds = xr.open_dataset(era5_path)

    try:
        # =====================================================
        # VALIDATE VARIABLES
        # =====================================================
        if "u10" not in ds or "v10" not in ds:
            raise Exception("❌ ERA5 missing u10/v10")

        # =====================================================
        # FIX LONGITUDE (0–360 → -180–180 IF NEEDED)
        # =====================================================
        if ds.longitude.max() > 180:
            ds = ds.assign_coords(
                longitude=(((ds.longitude + 180) % 360) - 180)
            ).sortby("longitude")

        # =====================================================
        # SUBSET TO SAR BBOX (CRITICAL FIX)
        # =====================================================
        ds = ds.sel(
            latitude=slice(max_lat, min_lat),   # ERA5 is descending
            longitude=slice(min_lon, max_lon)
        )

        # =====================================================
        # EXTRACT VARIABLES
        # =====================================================
        u10 = ds["u10"]
        v10 = ds["v10"]

        # TIME AVERAGE (if time exists)
        if "time" in u10.dims:
            u10 = u10.mean(dim="time")
            v10 = v10.mean(dim="time")

        u10 = np.nan_to_num(u10.values)
        v10 = np.nan_to_num(v10.values)

        wind_speed = np.sqrt(u10**2 + v10**2).astype(np.float32)

        lats = ds["latitude"].values
        lons = ds["longitude"].values

        # =====================================================
        # FIX LAT ORDER IF REQUIRED
        # =====================================================
        if lats[0] < lats[-1]:
            wind_speed = wind_speed[::-1, :]
            lats = lats[::-1]

        # =====================================================
        # DEBUG OUTPUT
        # =====================================================
        print("\n===== ERA5 GRID CHECK (AFTER CROPPING) =====")
        print("ERA5 lat range:", float(lats.min()), float(lats.max()))
        print("ERA5 lon range:", float(lons.min()), float(lons.max()))
        print("Wind range:", float(np.min(wind_speed)), "to", float(np.max(wind_speed)))

        # =====================================================
        # SAFETY CHECK
        # =====================================================
        if np.max(wind_speed) > 80:
            raise Exception("❌ Invalid ERA5 values detected")

        print("✅ ERA5 wind grid ready")

        return wind_speed, lats, lons

    finally:
        ds.close()