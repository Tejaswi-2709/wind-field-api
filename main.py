import json
import traceback
import numpy as np
import pandas as pd
import os

from pipeline import run_pipeline
from utils import plot_snap_style_wind
from validation import compute_metrics


# =====================================================
# MAIN JOB RUNNER
# =====================================================
def run_job(date, bbox, save_debug=False):

    try:

        # =================================================
        # CREATE OUTPUT FOLDER
        # =================================================

        output_dir = "outputs"

        os.makedirs(output_dir, exist_ok=True)

        # =================================================
        # RUN PIPELINE
        # =================================================

        result_df = run_pipeline(date, bbox)

        if result_df is None or result_df.empty:

            raise Exception("Pipeline returned empty dataframe")

        print(f"\n✅ Final rows: {len(result_df)}")

        # =================================================
        # GENERATE WIND PLOT
        # =================================================

        required_cols = {
            "lat",
            "lon",
            "sar_speed",
            "u",
            "v"
        }

        plot_path = None

        if required_cols.issubset(result_df.columns):

            plot_path = os.path.join(
                output_dir,
                f"wind_field_{date}.png"
            )

            plot_snap_style_wind(
                result_df,
                save_path=plot_path
            )

            print(f"✅ Plot saved: {plot_path}")

        else:

            print("⚠️ Missing plotting columns")

        # =================================================
        # COMPUTE VALIDATION METRICS
        # =================================================

        metrics = compute_metrics(result_df)

        print("\n===== METRICS =====")
        print(metrics)


        # =================================================
        # SAVE VECTOR CSV
        # =================================================

        csv_path = os.path.join(
            output_dir,
            f"wind_vectors_{date}.csv"
        )

        result_df.to_csv(
            csv_path,
            index=False
        )

        print(f"✅ CSV saved: {csv_path}")

        # =================================================
        # OPTIONAL DEBUG SAVE
        # =================================================

        if save_debug:

            debug_path = os.path.join(
                output_dir,
                "debug_output.csv"
            )

            result_df.to_csv(
                debug_path,
                index=False
            )

            print(f"✅ Debug saved: {debug_path}")

        # =================================================
        # CONVERT GEOMETRY
        # =================================================

        if "geometry" in result_df.columns:

            result_df["longitude"] = result_df["geometry"].apply(
                lambda p: p.x if p is not None else None
            )

            result_df["latitude"] = result_df["geometry"].apply(
                lambda p: p.y if p is not None else None
            )

            result_df = result_df.drop(
                columns=["geometry"]
            )

        # =================================================
        # CLEAN INVALID VALUES
        # =================================================

        result_df = result_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        result_df = result_df.where(
            pd.notnull(result_df),
            None
        )

        # =================================================
        # FINAL API RESPONSE
        # =================================================

        response = {

            "status": "success",

            "summary": {

                "rows": int(len(result_df)),

                "mae": round(metrics["mae"], 3)
                if metrics["mae"] is not None else None,

                "rmse": round(metrics["rmse"], 3)
                if metrics["rmse"] is not None else None,

                "bias": round(metrics["bias"], 3)
                if metrics["bias"] is not None else None
            },

            "plot_path": plot_path,

            "csv_path": csv_path
        }

        return response

    # =================================================
    # ERROR HANDLING
    # =================================================

    except Exception as e:

        print("\n❌ PIPELINE FAILED\n")

        print(str(e))

        return {

            "status": "failed",

            "error": str(e),

            "trace": traceback.format_exc()
        }


# =====================================================
# LOCAL TEST
# =====================================================

if __name__ == "__main__":

    date = ("2025-06-14")

    bbox = [68.0, 20.0, 72.5, 24.0]

    response = run_job(
        date=date,
        bbox=bbox,
        save_debug=True
    )

    print("\n===== RESPONSE =====\n")

    print(
        json.dumps(
            response,
            indent=2
        )
    )
