import os
import subprocess
import pandas as pd



from landmask import remove_land_points
from downloader import load_registry, save_registry
from era5 import load_era5_wind_new,get_era5_file
from downloader import get_sentinel_safe,RAW_DIR
from utils import build_safe_graph,snap_to_dataframe
from validation import match_sar_era5,compute_metrics
import os
# =====================================================
# SAFE SNAP RUNNER (FIXED CONTRACT)
# =====================================================
def run_snap_for_safe(product_info, bbox):

    safe_id = product_info.get("safe_id")
    safe_path = product_info.get("safe_path")
    output_dim = product_info.get("dim_path")
    cached = product_info.get("cached", False)

    xml_file = os.path.join("graphs", f"{safe_id}.xml")
    os.makedirs("graphs", exist_ok=True)

    # =================================================
    # CASE 1: DIM CACHE HIT
    # =================================================
    if cached and output_dim and os.path.exists(output_dim):
        print(f"✅ Using cached DIM: {safe_id}")
        return output_dim

    # =================================================
    # CASE 2: SAFE REQUIRED FOR SNAP
    # =================================================
    if not safe_path or not os.path.exists(safe_path):
        raise Exception(
            f"❌ SAFE missing for SNAP: {safe_id} | cached={cached}"
        )

    # =================================================
    # BUILD SNAP GRAPH
    # =================================================
    build_safe_graph(
        safe_path,
        bbox,
        output_xml=xml_file
    )

    print(f"🚀 Running SNAP GPT: {safe_id}")

    result = subprocess.run(
        [
            "gpt",
            xml_file,
            f"-Poutput={output_dim}"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"❌ SNAP failed: {safe_id}")

    if not os.path.exists(output_dim):
        raise Exception(f"❌ DIM not created: {safe_id}")

    print(f"✅ SNAP completed: {safe_id}")

    # SAFE cleanup (optional but safe)
    try:
        if safe_path and os.path.exists(safe_path):
            import shutil
            shutil.rmtree(safe_path)
    except Exception as e:
        print(f"⚠ SAFE cleanup failed: {e}")

    return output_dim


# =====================================================
# TRUE SAR EXTENT
# =====================================================
def get_sar_extent(df, margin=1.0):

    return [
        float(df["lon"].min()) - margin,
        float(df["lat"].min()) - margin,
        float(df["lon"].max()) + margin,
        float(df["lat"].max()) + margin,
    ]


# =====================================================
# MAIN PIPELINE (FULLY STABLE)
# =====================================================
def run_pipeline(date, bbox):

    print("\n===== AOI PIPELINE START =====\n")
    print("===== INPUT BBOX =====")
    print(bbox)

    # =================================================
    # GET PRODUCTS
    # =================================================
    response = get_sentinel_safe(date, bbox)
    safe_list = response.get("products", [])

    if not safe_list:
        raise Exception("❌ No SAFE products found")

    print(f"✅ Products received: {len(safe_list)}")

    all_dfs = []

    # =================================================
    # PROCESS EACH PRODUCT
    # =================================================
    for i, product_info in enumerate(safe_list):

        safe_id = product_info.get("safe_id")

        print(f"\n🧩 SAFE {i+1}/{len(safe_list)}")
        print(f"📍 Product: {safe_id}")

        try:

            # =================================================
            # SNAP PROCESS
            # =================================================
            dim_file = run_snap_for_safe(product_info, bbox)

            print(f"📍 DIM used: {dim_file}")

            # =================================================
            # UPDATE REGISTRY
            # =================================================
            registry = load_registry()
            registry[product_info["product_id"]] = {
                "dim_path": dim_file,
                "safe_id": safe_id
            }
            save_registry(registry)

            # =================================================
            # SNAP OUTPUT → DF
            # =================================================
            df = snap_to_dataframe(dim_file)

            if df is None or len(df) < 10:
                print("⚠️ Too few SAR points")
                continue

            # =================================================
            # FILTERING
            # =================================================
            df = df[
                (df["sar_speed"] > 0) &
                (df["sar_speed"] < 40) &
                (df["sar_dir"] >= 0) &
                (df["sar_dir"] <= 360)
            ]

            if df.empty:
                print("⚠️ Empty after filtering")
                continue

            # =================================================
            # LAND MASK
            # =================================================
            df = remove_land_points(df)

            if df.empty:
                print("⚠️ All points removed by landmask")
                continue

            df["safe_id"] = safe_id
            all_dfs.append(df)

            print(f"🌊 Ocean points: {len(df)}")

        except Exception as e:
            print(f"⚠️ FAILED SAFE: {safe_id}")
            print(e)
            continue

    # =================================================
    # FINAL CHECK
    # =================================================
    if not all_dfs:
        raise Exception("❌ No valid SAR data after processing")

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n📊 Total SAR points: {len(df)}")

    # =================================================
    # SAR EXTENT FOR ERA5
    # =================================================
    sar_bbox = get_sar_extent(df, margin=1.0)

    print("\n===== TRUE SAR EXTENT =====")
    print(sar_bbox)

    # =================================================
    # ERA5
    # =================================================
    era5_path = get_era5_file(date=date, bbox=sar_bbox)
    wind_speed, era5_lats, era5_lons = load_era5_wind_new(era5_path,sar_bbox)

    era5_data = {
        "speed": wind_speed,
        "lat": era5_lats,
        "lon": era5_lons
    }

    print("✅ ERA5 ready")

    # =================================================
    # MATCHING
    # =================================================
    print("\n===== MATCHING SAR WITH ERA5 =====")

    result_df = match_sar_era5(df, era5_data)

    if result_df.empty:
        raise Exception("❌ No matched SAR-ERA5 points")

    # =================================================
    # METRICS
    # =================================================
    print("\n===== VALIDATION METRICS =====")

    metrics = compute_metrics(result_df)
    print(metrics)

    # =================================================
    # SAVE OUTPUT
    # =================================================
    out_csv = f"final_comparison_{date}.csv"
    result_df.to_csv(out_csv, index=False)

    print(f"\n📁 Saved: {out_csv}")
    print(f"✅ Final rows: {len(result_df)}")

    return result_df