import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib
matplotlib.use("Agg")

# =========================
# OPTIONAL CARTOPY
# =========================
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except:
    CARTOPY_AVAILABLE = False


# =====================================================
# BASIC HELPERS
# =====================================================

def validate_safe_structure(safe_path):
    return os.path.exists(safe_path) and safe_path.endswith(".SAFE")


def bbox_to_wkt(bbox):
    minx, miny, maxx, maxy = bbox
    return f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"


# =====================================================
# SNAP GRAPH BUILDER (UNCHANGED LOGIC)
# =====================================================

def build_safe_graph(safe_path, bbox, output_xml="wind_graph.xml"):
    if not validate_safe_structure(safe_path):
        raise Exception("❌ Invalid SAFE path")

    graph = ET.Element("graph", id="WindGraph")
    ET.SubElement(graph, "version").text = "1.0"

    read = ET.SubElement(graph, "node", id="Read")
    ET.SubElement(read, "operator").text = "Read"
    params = ET.SubElement(read, "parameters")
    ET.SubElement(params, "file").text = safe_path

    subset = ET.SubElement(graph, "node", id="Subset")
    ET.SubElement(subset, "operator").text = "Subset"
    subset_sources = ET.SubElement(subset, "sources")
    ET.SubElement(subset_sources, "sourceProduct", refid="Read")

    subset_params = ET.SubElement(subset, "parameters")
    ET.SubElement(subset_params, "geoRegion").text = bbox_to_wkt(bbox)
    ET.SubElement(subset_params, "copyMetadata").text = "true"

    def add_node(name, operator, source):
        node = ET.SubElement(graph, "node", id=name)
        ET.SubElement(node, "operator").text = operator
        src = ET.SubElement(node, "sources")
        ET.SubElement(src, "sourceProduct", refid=source)
        ET.SubElement(node, "parameters")
        return node

    add_node("Apply-Orbit-File", "Apply-Orbit-File", "Subset")
    add_node("ThermalNoiseRemoval", "ThermalNoiseRemoval", "Apply-Orbit-File")
    add_node("Remove-GRD-Border-Noise", "Remove-GRD-Border-Noise", "ThermalNoiseRemoval")

    calib = ET.SubElement(graph, "node", id="Calibration")
    ET.SubElement(calib, "operator").text = "Calibration"
    calib_sources = ET.SubElement(calib, "sources")
    ET.SubElement(calib_sources, "sourceProduct", refid="Remove-GRD-Border-Noise")

    calib_params = ET.SubElement(calib, "parameters")
    ET.SubElement(calib_params, "selectedPolarisations").text = "VV"
    ET.SubElement(calib_params, "outputSigmaBand").text = "true"

    add_node("Speckle-Filter", "Speckle-Filter", "Calibration")

    wind = ET.SubElement(graph, "node", id="Wind-Field-Estimation")
    ET.SubElement(wind, "operator").text = "Wind-Field-Estimation"

    wind_sources = ET.SubElement(wind, "sources")
    ET.SubElement(wind_sources, "sourceProduct", refid="Speckle-Filter")

    wind_params = ET.SubElement(wind, "parameters")
    ET.SubElement(wind_params, "sourceBands").text = "Sigma0_VV"
    ET.SubElement(wind_params, "windowSizeInKm").text = "20.0"

    write = ET.SubElement(graph, "node", id="Write")
    ET.SubElement(write, "operator").text = "Write"

    write_sources = ET.SubElement(write, "sources")
    ET.SubElement(write_sources, "sourceProduct", refid="Wind-Field-Estimation")

    write_params = ET.SubElement(write, "parameters")
    ET.SubElement(write_params, "file").text = "${output}"
    ET.SubElement(write_params, "formatName").text = "BEAM-DIMAP"

    ET.ElementTree(graph).write(output_xml)

    print(f"✅ SNAP graph created: {output_xml}")
    return output_xml

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from scipy.interpolate import (
    griddata,
    RegularGridInterpolator
)

import matplotlib.pyplot as plt


# =====================================================
# READ SNAP HDR
# =====================================================

def _read_snap_shape(hdr_path):

    with open(hdr_path, "r") as f:
        text = f.read()

    lines = re.search(r"lines\s*=\s*(\d+)", text)
    samples = re.search(r"samples\s*=\s*(\d+)", text)

    if not lines or not samples:
        raise Exception("❌ Could not read HDR dimensions")

    lines = int(lines.group(1))
    samples = int(samples.group(1))

    return lines, samples


# =====================================================
# LOAD TIE POINT GRIDS
# =====================================================

def load_tie_point_grids(dim_path):

    data_folder = dim_path.replace(".dim", ".data")

    grid_folder = os.path.join(
        data_folder,
        "tie_point_grids"
    )

    lat_file = os.path.join(grid_folder, "latitude.img")
    lon_file = os.path.join(grid_folder, "longitude.img")

    lat_hdr = os.path.join(grid_folder, "latitude.hdr")

    if not os.path.exists(lat_file):
        raise Exception("❌ latitude.img missing")

    if not os.path.exists(lon_file):
        raise Exception("❌ longitude.img missing")

    lines, samples = _read_snap_shape(lat_hdr)

    # SNAP stores big-endian float32
    lat = np.fromfile(
        lat_file,
        dtype=">f4"
    ).astype(np.float32)

    lon = np.fromfile(
        lon_file,
        dtype=">f4"
    ).astype(np.float32)

    lat = lat.reshape(lines, samples)
    lon = lon.reshape(lines, samples)

    print(f"✅ Tie-point grid shape: {lat.shape}")

    return lat, lon


# =====================================================
# PARSE WIND CSV
# =====================================================

def extract_wind_csv(data_folder):

    if data_folder.endswith(".dim"):
        data_folder = data_folder.replace(".dim", ".data")

    wind_csv = os.path.join(
        data_folder,
        "vector_data",
        "WindField.csv"
    )

    if not os.path.exists(wind_csv):
        raise Exception("❌ WindField.csv missing")

    df = pd.read_csv(
        wind_csv,
        sep=None,
        engine="python",
        skiprows=1
    )

    # clean headers
    df.columns = [
        c.strip().lower().split(":")[0]
        for c in df.columns
    ]

    rename_map = {
        "speed": "sar_speed",
        "heading": "sar_dir"
    }

    df = df.rename(columns=rename_map)

    required = [
        "sar_speed",
        "sar_dir",
        "geometry"
    ]

    for c in required:
        if c not in df.columns:
            raise Exception(f"❌ Missing column: {c}")

    df["sar_speed"] = pd.to_numeric(
        df["sar_speed"],
        errors="coerce"
    )

    df["sar_dir"] = pd.to_numeric(
        df["sar_dir"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["sar_speed", "sar_dir"]
    )

    print(f"✅ Raw SAR vectors: {len(df)}")

    return df


# =====================================================
# IMAGE COORDS -> LAT/LON
# =====================================================

def assign_lat_lon_from_tiepoint(
    df,
    lat_grid,
    lon_grid
):

    if "geometry" not in df.columns:
        raise Exception("❌ geometry column missing")

    grid_rows, grid_cols = lat_grid.shape

    # -------------------------------------------------
    # parse image coordinates from POINT(x y)
    # -------------------------------------------------

    def parse_pixel(g):

        try:

            if not isinstance(g, str):
                return None

            if "POINT" not in g:
                return None

            vals = (
                g.replace("POINT", "")
                 .replace("(", "")
                 .replace(")", "")
                 .split()
            )

            x = float(vals[0])
            y = float(vals[1])

            return x, y

        except:
            return None

    parsed = df["geometry"].apply(parse_pixel)

    mask = parsed.notna()

    df = df[mask].copy()
    parsed = parsed[mask]

    pixel_x = parsed.apply(lambda p: p[0]).values
    pixel_y = parsed.apply(lambda p: p[1]).values

    # -------------------------------------------------
    # IMPORTANT:
    # tie-point grids are coarse grids
    # so we normalize coordinates
    # -------------------------------------------------

    x_max = np.max(pixel_x)
    y_max = np.max(pixel_y)

    if x_max <= 0 or y_max <= 0:
        raise Exception("❌ Invalid geometry coordinates")

    scaled_x = (
        pixel_x / x_max
    ) * (grid_cols - 1)

    scaled_y = (
        pixel_y / y_max
    ) * (grid_rows - 1)

    # -------------------------------------------------
    # build interpolators
    # -------------------------------------------------

    rows = np.arange(grid_rows)
    cols = np.arange(grid_cols)

    lat_interp = RegularGridInterpolator(
        (rows, cols),
        lat_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    lon_interp = RegularGridInterpolator(
        (rows, cols),
        lon_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    sample_points = np.column_stack([
        scaled_y,
        scaled_x
    ])

    lat_vals = lat_interp(sample_points)
    lon_vals = lon_interp(sample_points)

    df["lat"] = lat_vals
    df["lon"] = lon_vals

    # -------------------------------------------------
    # sanity filter
    # -------------------------------------------------

    df = df[
        df["lat"].between(-90, 90)
    ]

    df = df[
        df["lon"].between(-180, 180)
    ]

    print(f"✅ Geolocated vectors: {len(df)}")

    return df


# =====================================================
# WIND COMPONENTS
# =====================================================

def add_uv_components(df):

    theta = np.deg2rad(df["sar_dir"])

    # meteorological convention
    df["u"] = -df["sar_speed"] * np.sin(theta)
    df["v"] = -df["sar_speed"] * np.cos(theta)

    return df


# =====================================================
# SNAP -> FINAL DATAFRAME
# =====================================================

def snap_to_dataframe(dim_file):

    data_folder = dim_file.replace(
        ".dim",
        ".data"
    )

    if not os.path.exists(data_folder):
        raise Exception("❌ SNAP .data missing")

    # load csv
    df = extract_wind_csv(data_folder)

    # load tie-point grids
    lat_grid, lon_grid = load_tie_point_grids(dim_file)

    # geolocate vectors
    df = assign_lat_lon_from_tiepoint(
        df,
        lat_grid,
        lon_grid
    )

    # add wind vectors
    df = add_uv_components(df)

    # create geodataframe
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["lon"],
            df["lat"]
        ),
        crs="EPSG:4326"
    )

    print(f"✅ Final SAR vectors: {len(gdf)}")

    return gdf

# =====================================================
# WIND FIELD PLOT
# =====================================================
def plot_snap_style_wind(df,save_path=None):

    import numpy as np
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from scipy.interpolate import griddata
    from global_land_mask import globe

    # =========================================
    # INPUT DATA
    # =========================================

    lon = df["lon"].values
    lat = df["lat"].values

    u = df["u"].values
    v = df["v"].values

    speed = df["sar_speed"].values

    # =========================================
    # CREATE HIGH-RESOLUTION GRID
    # =========================================

    grid_lon = np.linspace(
        lon.min(),
        lon.max(),
        120
    )

    grid_lat = np.linspace(
        lat.min(),
        lat.max(),
        120
    )

    grid_lon, grid_lat = np.meshgrid(
        grid_lon,
        grid_lat
    )

    # =========================================
    # INTERPOLATION (SMOOTH)
    # =========================================

    grid_u = griddata(
        (lon, lat),
        u,
        (grid_lon, grid_lat),
        method="cubic"
    )

    grid_v = griddata(
        (lon, lat),
        v,
        (grid_lon, grid_lat),
        method="cubic"
    )

    grid_speed = griddata(
        (lon, lat),
        speed,
        (grid_lon, grid_lat),
        method="cubic"
    )

    # =========================================
    # FILL EMPTY OCEAN GAPS
    # =========================================

    u_fill = griddata(
        (lon, lat),
        u,
        (grid_lon, grid_lat),
        method="nearest"
    )

    v_fill = griddata(
        (lon, lat),
        v,
        (grid_lon, grid_lat),
        method="nearest"
    )

    speed_fill = griddata(
        (lon, lat),
        speed,
        (grid_lon, grid_lat),
        method="nearest"
    )

    grid_u = np.where(
        np.isnan(grid_u),
        u_fill,
        grid_u
    )

    grid_v = np.where(
        np.isnan(grid_v),
        v_fill,
        grid_v
    )

    grid_speed = np.where(
        np.isnan(grid_speed),
        speed_fill,
        grid_speed
    )

    # =========================================
    # LAND MASK AFTER INTERPOLATION
    # VERY IMPORTANT
    # =========================================

    land_mask = globe.is_land(
        grid_lat,
        grid_lon
    )

    grid_u[land_mask] = np.nan
    grid_v[land_mask] = np.nan
    grid_speed[land_mask] = np.nan

    # =========================================
    # NORMALIZE VECTORS
    # EQUAL ARROW SIZE
    # =========================================

    magnitude = np.sqrt(
        grid_u**2 + grid_v**2
    )

    u_norm = grid_u / (magnitude + 1e-6)
    v_norm = grid_v / (magnitude + 1e-6)

    # =========================================
    # FIGURE
    # =========================================

    fig = plt.figure(figsize=(14, 10))

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    # =========================================
    # MAP FEATURES
    # =========================================

    ax.set_facecolor("#9bb6d8")

    ax.add_feature(
        cfeature.LAND,
        facecolor="lightgray",
        zorder=11
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=1.2,
        zorder=11
    )

    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
        alpha=0.5
    )

    # =========================================
    # EXTENT
    # =========================================

    ax.set_extent([
        lon.min(),
        lon.max(),
        lat.min(),
        lat.max()
    ])

    # =========================================
    # WIND SPEED BACKGROUND
    # =========================================

    contour = ax.contourf(
        grid_lon,
        grid_lat,
        grid_speed,

        levels=np.linspace(
            0,
            20,
            21
        ),

        cmap="turbo",

        extend="max",

        transform=ccrs.PlateCarree(),

        zorder=1
    )

    # =========================================
    # REDUCE ARROW CONGESTION
    # =========================================

    step = 4

    # =========================================
    # WIND VECTORS
    # =========================================

    ax.quiver(

        grid_lon[::step, ::step],
        grid_lat[::step, ::step],

        u_norm[::step, ::step],
        v_norm[::step, ::step],

        color="black",

        scale=35,

        width=0.002,

        headwidth=3,
        headlength=4,

        pivot="middle",

        transform=ccrs.PlateCarree(),

        zorder=20
    )

    # =========================================
    # COLORBAR
    # =========================================

    cbar = plt.colorbar(
        contour,
        ax=ax,
        shrink=0.75,
        pad=0.04
    )

    cbar.set_label(
        "Wind Speed (m/s)",
        fontsize=14
    )

    # =========================================
    # GRIDLINES
    # =========================================

    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=0.5
    )

    gl.top_labels = False
    gl.right_labels = False

    # =========================================
    # TITLE
    # =========================================

    plt.title(
        "SAR Wind Field (Ocean Only)",
        fontsize=22,
        weight="bold"
    )

    # =========================================
    # TIGHT LAYOUT
    # =========================================

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"✅ Wind plot saved: {save_path}")


    plt.close()