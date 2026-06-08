import os
import requests
import zipfile
import shutil
import json

from datetime import datetime, timedelta
from shapely.geometry import box, shape
from dotenv import load_dotenv

# =====================================================
# ENV + REGISTRY
# =====================================================
load_dotenv()

PROCESSED_DB = "Data/processed_registry.json"


def load_registry():
    if not os.path.exists(PROCESSED_DB):
        return {}
    with open(PROCESSED_DB, "r") as f:
        return json.load(f)


def save_registry(db):
    with open(PROCESSED_DB, "w") as f:
        json.dump(db, f, indent=2)


# =====================================================
# ENV VARIABLES
# =====================================================
CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")


# =====================================================
# DIRECTORIES
# =====================================================
RAW_DIR = "Data/raw_safe"
ZIP_DIR = "Data/raw_downloads"
DIM_DIR = "Data/processed_dim"
CACHE_DIR = "Data/cache"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(DIM_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# =====================================================
# AUTH
# =====================================================
def get_access_token():
    url = (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/protocol/openid-connect/token"
    )

    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": CDSE_USERNAME,
        "password": CDSE_PASSWORD,
    }

    r = requests.post(url, data=data, timeout=60)

    if r.status_code != 200:
        raise Exception(f"Auth failed: {r.text}")

    return r.json()["access_token"]


# =====================================================
# BBOX
# =====================================================
def bbox_to_wkt(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox

    return (
        f"POLYGON(("
        f"{min_lon} {min_lat},"
        f"{max_lon} {min_lat},"
        f"{max_lon} {max_lat},"
        f"{min_lon} {max_lat},"
        f"{min_lon} {min_lat}"
        f"))"
    )


# =====================================================
# SEARCH
# =====================================================
def search_products(date, bbox, token):

    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    headers = {"Authorization": f"Bearer {token}"}

    start = f"{date}T00:00:00.000Z"
    end = f"{date}T23:59:59.999Z"

    polygon = bbox_to_wkt(bbox)

    query = (
        "Collection/Name eq 'SENTINEL-1' and "
        "contains(Name,'GRD') and "
        f"ContentDate/Start ge {start} and "
        f"ContentDate/Start le {end} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
    )

    params = {"$filter": query, "$top": 50}

    r = requests.get(url, headers=headers, params=params, timeout=120)

    if r.status_code != 200:
        raise Exception(f"Search failed: {r.text}")

    products = r.json().get("value", [])

    print(f"🔎 AOI products found: {len(products)}")

    return products


# =====================================================
# VALIDATION
# =====================================================
def is_valid_product(name):
    if not name:
        return False
    if "_COG" in name:
        return False
    if "IW_GRDH" not in name:
        return False
    if ".SAFE" not in name:
        return False
    return True


def rank_products(products, bbox):
    aoi = box(*bbox)

    scored = []
    seen = set()

    for p in products:
        pid = p.get("Id")
        if not pid or pid in seen:
            continue

        seen.add(pid)

        name = p.get("Name", "")
        if not is_valid_product(name):
            continue

        geom = p.get("GeoFootprint") or p.get("Footprint")
        if not geom:
            continue

        try:
            overlap = shape(geom).intersection(aoi).area
        except Exception:
            overlap = 0

        scored.append((overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


# =====================================================
# DOWNLOAD URL
# =====================================================
def get_download_url(product_id):
    return (
        f"https://zipper.dataspace.copernicus.eu/"
        f"odata/v1/Products({product_id})/$value"
    )


# =====================================================
# MAIN DOWNLOAD FUNCTION (FIXED CONTRACT)
# =====================================================
def download_and_extract(product, token):

    name = product.get("Name", "")
    product_id = product.get("Id")

    safe_id = name.replace(".SAFE", "")

    registry = load_registry()

    # IMPORTANT: ALWAYS DEFINE VARIABLES
    safe_path = None
    dim_path = os.path.join(DIM_DIR, f"{safe_id}_wind.dim")

    cache_file = os.path.join(CACHE_DIR, f"{product_id}.done")

    # =====================================================
    # CACHE HIT → NO SAFE PATH RETURNED (IMPORTANT FIX)
    # =====================================================
    if os.path.exists(cache_file):
        print(f"⚡ Cached product (no SAFE available): {product_id}")

        return {
            "safe_id": safe_id,
            "safe_path": None,   # IMPORTANT: explicitly None
            "dim_path": dim_path if os.path.exists(dim_path) else None,
            "product_name": name,
            "product_id": product_id,
            "cached": True,
            "has_safe": False
        }

    # =====================================================
    # REGISTRY HIT
    # =====================================================
    if product_id in registry:
        print(f"⚡ Registry hit: {safe_id}")

        return {
            "safe_id": safe_id,
            "safe_path": None,
            "dim_path": registry[product_id]["dim_path"],
            "product_name": name,
            "product_id": product_id,
            "cached": True,
            "has_safe": False
        }

    # =====================================================
    # DIM CACHE (REAL USABLE OUTPUT)
    # =====================================================
    if os.path.exists(dim_path):
        print(f"✅ DIM cache hit: {safe_id}")

        return {
            "safe_id": safe_id,
            "safe_path": None,
            "dim_path": dim_path,
            "product_name": name,
            "product_id": product_id,
            "cached": True,
            "has_safe": False
        }

    # =====================================================
    # DOWNLOAD PATHS
    # =====================================================
    zip_path = os.path.join(ZIP_DIR, f"{safe_id}.zip")
    safe_path = os.path.join(RAW_DIR, name)

    url = get_download_url(product_id)
    headers = {"Authorization": f"Bearer {token}"}

    print(f"⬇️ Downloading: {safe_id}")

    r = requests.get(url, headers=headers, stream=True, timeout=(30, 300))

    if r.status_code != 200:
        print(f"❌ Download failed: {safe_id}")
        return None

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    # =====================================================
    # EXTRACT
    # =====================================================
    print(f"📦 Extracting: {safe_id}")

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # =====================================================
    # SAVE CACHE
    # =====================================================
    with open(cache_file, "w") as f:
        f.write("done")

    print(f"✅ SAFE ready: {safe_id}")
    print("📁 SAFE exists:", os.path.exists(safe_path))

    return {
        "safe_id": safe_id,
        "safe_path": safe_path,
        "dim_path": dim_path,
        "product_name": name,
        "product_id": product_id,
        "cached": False,
        "has_safe": True
    }


# =====================================================
# CLEANUP
# =====================================================
def cleanup_safe(safe_path):
    if safe_path and os.path.exists(safe_path):
        shutil.rmtree(safe_path)
        print(f"🗑 SAFE deleted: {safe_path}")


# =====================================================
# MAIN PIPELINE ENTRY
# =====================================================
def get_sentinel_safe(date, bbox, max_products=3):

    print("\n===== FAST AOI DOWNLOADER =====")

    token = get_access_token()

    current_date = datetime.strptime(date, "%Y-%m-%d")

    products = []
    used_date = None

    for _ in range(5):

        d = current_date.strftime("%Y-%m-%d")
        print(f"📅 Searching {d}")

        raw = search_products(d, bbox, token)

        if raw:
            products = rank_products(raw, bbox)
            used_date = d
            break

        current_date -= timedelta(days=1)

    if not products:
        raise Exception("No AOI products found")

    products = products[:max_products]

    print(f"🚀 Using top {len(products)} products")

    results = []

    for p in products:
        if "_COG" in p.get("Name", ""):
            continue

        res = download_and_extract(p, token)

        if res:
            results.append(res)

    print(f"✅ Ready products: {len(results)}")

    return {
        "used_date": used_date,
        "count": len(results),
        "products": results
    }