# app.py
# Run with: streamlit run app.py

import re, json, numpy as np, pandas as pd, geopandas as gpd, streamlit as st, pydeck as pdk
from shapely.geometry import box
from datetime import datetime

# =========================
# CONFIG: Update these paths
# =========================
CSV_PATH  = r"C:\Users\sshaban\Music\Project I\final_for_stearmilit.csv"
SAL_PATH  = r"C:\Users\sshaban\Music\Project I\SAL_2021_AUST_GDA2020\SAL_2021_AUST_GDA2020.shp"
PERTH_BBOX = (115.50, -32.60, 116.20, -31.40)

# =========================
# COLORS & STYLES
# =========================
R_PALETTE = {
    "R10":[205,236,205,190], "R12.5":[192,229,192,190], "R15":[177,221,177,190],
    "R17.5":[162,212,162,190], "R20":[147,204,147,190], "R25":[126,192,126,190],
    "R30":[106,182,106,190], "R40":[84,170,84,190], "R50":[66,158,66,190],
    "R60":[49,146,49,190], "R80":[38,132,38,190], "R100":[24,118,24,190],
    "R-AC0":[90,180,90,190]
}
DEFAULT_GREY = [210,214,218,170]

def _auto_green_for_numeric(rnum: float):
    light = (205,236,205); dark = (24,118,24)
    t = float(np.clip((rnum - 10.0) / 90.0, 0.0, 1.0))
    lerp = lambda a,b: int(round(a + (b - a) * t))
    return [lerp(light[0], dark[0]), lerp(light[1], dark[1]), lerp(light[2], dark[2]), 190]

def bf_color_from_pct(p):
    if pd.isna(p): p = 0.0
    p = float(np.clip(p, 0, 100))
    if p == 0: return [230,230,230,160]
    elif p < 5: return [255,229,204,200]
    elif p < 20: return [255,204,153,210]
    elif p < 40: return [255,153,51,220]
    elif p < 60: return [255,127,14,230]
    else: return [204,85,0,240]

def normalize_suburb(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    s = str(s).strip()
    s = re.sub(r"\s*\(WA\)$", "", s, flags=re.I)
    s = re.sub(r"^Balcata$", "Balcatta", s, flags=re.I)
    return s

# ---------- Helpers for dynamic ICSEA colouring & legends ----------
def _hex(rgb_or_rgba):
    r,g,b = rgb_or_rgba[:3]
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

def _lerp(a, b, t): return a + (b - a) * t
def _lerp_rgb(c1, c2, t):
    return [int(round(_lerp(c1[0], c2[0], t))),
            int(round(_lerp(c1[1], c2[1], t))),
            int(round(_lerp(c1[2], c2[2], t))), 220]

def make_step_bins(vmin, vmax, step=100):
    if pd.isna(vmin) or pd.isna(vmax): return []
    vmin, vmax = float(min(vmin, vmax)), float(max(vmin, vmax))
    edges = [vmin]; cur = vmin
    while cur + step < vmax:
        cur += step; edges.append(cur)
    if edges[-1] != vmax: edges.append(vmax)
    return edges

def make_orange_yellow_green_palette(n):
    if n <= 1: return [[255,165,0,220]]
    start, mid, end = [255,165,0,220], [255,255,0,220], [24,118,24,220]
    colors = []
    for i in range(n):
        t = i/(n-1) if n>1 else 0
        colors.append(_lerp_rgb(start, mid, t/0.5) if t<=0.5 else _lerp_rgb(mid, end, (t-0.5)/0.5))
    return colors

def assign_bin_color(series_vals):
    vals = pd.to_numeric(series_vals, errors="coerce").dropna()
    if vals.empty:
        return None, []
    vmin, vmax = float(vals.min()), float(vals.max())
    if np.isclose(vmin, vmax):
        edges = [vmin, vmax]
        palette = make_orange_yellow_green_palette(1)
        def colour_for(_): return palette[0]
        legend = [(f"{int(round(vmin))}", palette[0])]
        return {"edges": edges, "palette": palette, "colour_fn": colour_for}, legend

    edges = make_step_bins(vmin, vmax, step=100)
    if len(edges) < 2: edges = [vmin, vmax]

    n_bins = len(edges) - 1
    palette = make_orange_yellow_green_palette(n_bins)

    legend = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        legend.append((f"{int(round(lo))} – {int(round(hi))}", palette[i]))

    def colour_for(v):
        if pd.isna(v): return [200,200,200,160]
        v = float(v)
        for i in range(n_bins-1):
            if edges[i] <= v < edges[i+1]:
                return palette[i]
        return palette[-1]

    return {"edges": edges, "palette": palette, "colour_fn": colour_for}, legend

def render_legend(title, legend_items):
    if not legend_items:
        st.caption(f"{title}: no data"); return
    parts = []
    for label, rgba in legend_items:
        parts.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:10px;margin-bottom:6px;'>"
            f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #666;background:{_hex(rgba)};margin-right:6px;'></span>"
            f"<span style='font-size:0.9rem;'>{label}</span></span>"
        )
    st.markdown(f"<div style='margin-top:4px'><strong style='font-size:0.9rem'>{title}:</strong> {''.join(parts)}</div>", unsafe_allow_html=True)

# ---------- Price colouring ----------
def price_color(v, bins):
    if pd.isna(v): return [200,200,200,160]
    if v < bins[0]:   return [204,229,255,220]
    elif v < bins[1]: return [153,204,255,220]
    elif v < bins[2]: return [51,153,255,230]
    else:             return [0,102,204,230]

# ---------- R-code helpers ----------
def rkey(rc):
    if pd.isna(rc): return None
    s = str(rc).strip().upper().replace(" ", "")
    s = re.split(r"[/-]", s)[0]
    if re.fullmatch(r"\d+(\.\d+)?", s): s = f"R{s}"
    s = re.sub(r"^R-?", "R", s)
    s = re.sub(r"\.0+$", "", s)
    if not s.startswith("R"): s = f"R{s}"
    return s

def color_for_key(k: str):
    if k is None: return DEFAULT_GREY
    if k in R_PALETTE: return R_PALETTE[k]
    m = re.search(r"R(\d+(?:\.\d+)?)$", k)
    if m: return _auto_green_for_numeric(float(m.group(1)))
    return DEFAULT_GREY

def to_geojson(gdf: gpd.GeoDataFrame, color_col: str) -> dict:
    feats = []
    for _, row in gdf.iterrows():
        try:
            geom = json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"]
        except Exception:
            continue
        props = {k: row[k] for k in gdf.columns if k != "geometry"}
        props["color"] = row[color_col]
        feats.append({"type":"Feature","geometry":geom,"properties":props})
    return {"type":"FeatureCollection","features":feats}

def view_for(gdf):
    if len(gdf)==0: return pdk.ViewState(latitude=-31.95, longitude=115.86, zoom=9.2)
    u = gdf.unary_union; c = u.centroid
    minx, miny, maxx, maxy = gdf.total_bounds
    span = max(maxx-minx, maxy-miny)
    zoom = 11 if span < 0.2 else 10 if span < 0.6 else 9.2
    return pdk.ViewState(latitude=float(c.y), longitude=float(c.x), zoom=zoom)

def make_map(gdf, color_col, tooltip_html, view_state):
    layer = pdk.Layer(
        "GeoJsonLayer", data=to_geojson(gdf, color_col),
        opacity=1.0, stroked=True, get_fill_color="properties.color",
        get_line_color=[120,120,120,140], lineWidthMinPixels=0.5, pickable=True)
    tooltip = {"html": tooltip_html, "style": {"backgroundColor":"white","color":"black"}}
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, height=400)

@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"Postcode":"string"})
    if "Suburb" in df.columns:
        df["Suburb"] = df["Suburb"].map(normalize_suburb)
    elif "Suburb_x" in df.columns:
        df["Suburb"] = df["Suburb_x"].map(normalize_suburb)
    if "Region" in df.columns:
        df["Region"] = df["Region"].astype(str).str.strip()
    if "Top_rcode" in df.columns:
        df["Top_rcode"] = df["Top_rcode"].astype(str).str.strip()
    if "Date_Sold" in df.columns:
        df["Date_Sold"] = pd.to_datetime(df["Date_Sold"], dayfirst=True, errors="coerce")
    for col in ["Price","Distance_to_CBD_km","Primary_School_ICSEA","Secondary_School_ICSEA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_final_csv(path): return load_csv(path)

@st.cache_data(show_spinner=False)
def load_sal_wa(sal_path: str, bbox) -> gpd.GeoDataFrame:
    sal = gpd.read_file(sal_path)
    if sal.crs is None: sal = sal.set_crs(7844)
    sal = sal[sal["STE_NAME21"].str.contains("Western Australia", case=False, na=False)].to_crs(4326)
    minx, miny, maxx, maxy = bbox
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(minx,miny,maxx,maxy)], crs=4326)
    sal = gpd.overlay(sal, bbox_gdf, how="intersection")
    sal["Suburb"] = sal["SAL_NAME21"].map(normalize_suburb)
    return sal[["Suburb","geometry"]]

# =========================
# APP
# =========================
st.set_page_config(page_title="ZoneWise Perth", layout="wide")
st.markdown("<h1 style='text-align:center; margin-top:0'>ZoneWise Perth</h1>", unsafe_allow_html=True)

df = load_final_csv(CSV_PATH)
sal = load_sal_wa(SAL_PATH, PERTH_BBOX)

# Use recent data for nicer slider defaults
df_recent = df[df.get("Date_Sold", pd.Timestamp("1900-01-01")) >= pd.Timestamp(2024,1,1)].copy()

# ---------- Compute all defaults BEFORE widgets ----------
if "Price" in df_recent.columns and df_recent["Price"].notna().sum():
    p_min = int(np.nanpercentile(df_recent["Price"], 1))
    p_max = int(np.nanpercentile(df_recent["Price"], 99))
else:
    p_min, p_max = 0, 3_000_000

regions_list = sorted([r for r in df_recent.get("Region", pd.Series(dtype=str)).dropna().unique()])

if "Distance_to_CBD_km" in df_recent.columns and df_recent["Distance_to_CBD_km"].notna().sum():
    d_min = float(np.nanmin(df_recent["Distance_to_CBD_km"]))
    d_max = float(np.nanmax(df_recent["Distance_to_CBD_km"]))
else:
    d_min, d_max = 0.0, 30.0
max_dist_default = float(round(min(25.0, d_max), 1))

def _icsea_bounds(col):
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if not vals.empty:
            return int(vals.min()), int(vals.max())
    return 0, 1500
p_lo, p_hi = _icsea_bounds("Primary_School_ICSEA")
s_lo, s_hi = _icsea_bounds("Secondary_School_ICSEA")

rcodes_all = sorted(
    set([rkey(x) for x in df.get("Top_rcode", pd.Series(dtype=str)).dropna().unique() if rkey(x)]),
    key=lambda s: float(re.search(r"R(\d+(?:\.\d+)?)", s).group(1)) if re.search(r"R(\d+(?:\.\d+)?)", s) else 0
)

suburbs_all = sorted(df["Suburb"].dropna().unique())
postcodes_all = sorted(df["Postcode"].dropna().unique())

# ---------- Clear-all mechanism (pre-widget reset) ----------
def _request_clear():
    st.session_state["__do_clear__"] = True

def _reset_all_to_defaults():
    st.session_state["budget"]             = (p_min, p_max)
    st.session_state["regions"]            = []
    st.session_state["max_dist"]           = max_dist_default
    st.session_state["rcodes"]             = []
    st.session_state["prim_icsea"]         = (p_lo, p_hi)
    st.session_state["sec_icsea"]          = (s_lo, s_hi)
    st.session_state["suburb_dropdown"]    = ""
    st.session_state["postcode_dropdown"]  = ""
    st.session_state["filters"]            = {}

if st.session_state.get("__do_clear__", False):
    _reset_all_to_defaults()
    st.session_state["__do_clear__"] = False
    st.rerun()

if "filters" not in st.session_state:
    st.session_state.filters = {}

# ---------------- Sidebar (applies to all tabs) ----------------
st.sidebar.title("Filters")

# ---- TOP buttons ----
top_c1, top_c2 = st.sidebar.columns(2)
with top_c1:
    apply_clicked = st.button("Apply Filters", type="primary", use_container_width=True)
with top_c2:
    st.button("Clear all", use_container_width=True, on_click=_request_clear)

# ===== GROUP 1 (AND) =====
st.sidebar.subheader("Group 1 (AND)")

budget = st.sidebar.slider(
    "What is your budget? (AUD)",
    min_value=p_min, max_value=p_max,
    value=st.session_state.get("budget", (p_min, p_max)),
    step=1000, key="budget"
)

pick_regions = st.sidebar.multiselect(
    "Which region(s) are you considering?",
    options=regions_list, default=st.session_state.get("regions", []),
    key="regions"
)

max_dist = st.sidebar.slider(
    "Maximum distance from CBD (km)",
    min_value=float(round(d_min,1)),
    max_value=float(round(d_max,1)),
    value=st.session_state.get("max_dist", max_dist_default),
    step=0.5, key="max_dist"
)

pick_rcodes = st.sidebar.multiselect(
    "R zoning (optional)",
    options=rcodes_all, default=st.session_state.get("rcodes", []),
    key="rcodes"
)

prim_icsea_range = st.sidebar.slider(
    "Primary School ICSEA range",
    min_value=p_lo, max_value=p_hi,
    value=st.session_state.get("prim_icsea", (p_lo, p_hi)),
    step=10, key="prim_icsea"
)
sec_icsea_range  = st.sidebar.slider(
    "Secondary School ICSEA range",
    min_value=s_lo, max_value=s_hi,
    value=st.session_state.get("sec_icsea", (s_lo, s_hi)),
    step=10, key="sec_icsea"
)

# ---- separator ----
st.sidebar.markdown("<div style='height:18px'></div><hr><div style='text-align:center; font-weight:600;'>— OR —</div><hr>", unsafe_allow_html=True)

# ===== GROUP 2 (OR): Suburb / Postcode =====
st.sidebar.subheader("Group 2 (OR)")
suburb_input  = st.sidebar.selectbox(
    "Select or type Suburb", [""] + suburbs_all,
    index=0, key="suburb_dropdown"
)
postcode_input = st.sidebar.selectbox(
    "Select Postcode", [""] + postcodes_all,
    index=0, key="postcode_dropdown"
)

# ------- APPLY / INIT -------
if apply_clicked or not st.session_state.filters:
    st.session_state.filters = {
        "budget": st.session_state["budget"],
        "regions": st.session_state["regions"],
        "max_dist": st.session_state["max_dist"],
        "rcodes": st.session_state["rcodes"],
        "prim_icsea": st.session_state["prim_icsea"],
        "sec_icsea": st.session_state["sec_icsea"],
        "suburb": st.session_state["suburb_dropdown"].strip(),
        "postcode": str(st.session_state["postcode_dropdown"]).strip(),
    }

F = st.session_state.filters

# ---------- filtering helpers ----------
def filter_range_with_nans(dfX, col, lo, hi):
    if col not in dfX.columns:
        return dfX
    s = pd.to_numeric(dfX[col], errors="coerce")
    mask = s.between(lo, hi, inclusive="both") | s.isna()
    return dfX[mask]

def apply_all_filters(df0: pd.DataFrame) -> pd.DataFrame:
    df1 = df0.copy()
    if "Date_Sold" in df1.columns:
        df1 = df1[df1["Date_Sold"] >= pd.Timestamp(2024,1,1)]

    if "Price" in df1.columns:
        df1 = df1[df1["Price"].between(F["budget"][0], F["budget"][1], inclusive="both")]
    if "Distance_to_CBD_km" in df1.columns:
        df1 = df1[df1["Distance_to_CBD_km"] <= F["max_dist"]]
    if F["regions"] and "Region" in df1.columns:
        df1["Region"] = df1["Region"].astype(str).str.strip()
        df1 = df1[df1["Region"].isin(F["regions"])]
    if F["rcodes"]:
        tmp = df1.copy()
        tmp["__rk"] = tmp.get("Top_rcode", pd.Series(index=tmp.index)).astype(str).str.strip().map(rkey)
        df1 = tmp[tmp["__rk"].isin(F["rcodes"])].drop(columns=["__rk"], errors="ignore")

    df1 = filter_range_with_nans(df1, "Primary_School_ICSEA",  F["prim_icsea"][0], F["prim_icsea"][1])
    df1 = filter_range_with_nans(df1, "Secondary_School_ICSEA",F["sec_icsea"][0],  F["sec_icsea"][1])

    sub = F.get("suburb", "")
    pc  = F.get("postcode", "")
    if sub or pc:
        s_sub = df1.get("Suburb", pd.Series("", index=df1.index)).astype(str)
        s_pc  = df1.get("Postcode", pd.Series("", index=df1.index)).astype(str)
        mask_sub = (s_sub == sub) if sub else False
        mask_pc  = (s_pc  == pc)  if pc  else False
        df1 = df1[ mask_sub | mask_pc ]

    return df1

# Optional count for debugging
try:
    _count = len(apply_all_filters(df))
    st.sidebar.caption(f"Matches: **{_count}** rows after filters")
except Exception:
    pass

# ---------------- Tabs (Info guide LAST) ----------------
tab1, tab2, tab3, tab4, tab_info = st.tabs(
    ["🏫 School", "🗺️ Geographic info", "💲 Pricing", "✅ Final suggestion", "ℹ️ Info guide"]
)

# ============ TAB 1: SCHOOL ============
with tab1:
    filt = apply_all_filters(df)

    st.subheader("School options for your selection")

    has_cols = {"Suburb","Postcode","Level","School"}
    if has_cols.issubset(filt.columns) and not filt.empty:
        primary_tbl = (filt[filt["Level"].str.contains("Primary", case=False, na=False)]
                       [["Suburb","Postcode","Level","School"]]
                       .dropna(subset=["School"]).drop_duplicates()
                       .sort_values(["Suburb","School"]))
        secondary_tbl = (filt[filt["Level"].str.contains("Secondary", case=False, na=False)]
                         [["Suburb","Postcode","Level","School"]]
                         .dropna(subset=["School"]).drop_duplicates()
                         .sort_values(["Suburb","School"]))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Primary")
            st.dataframe(primary_tbl.reset_index(drop=True), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### Secondary")
            st.dataframe(secondary_tbl.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.info("No school rows for the current filters.")

    # --- School performance maps (ICSEA) ---
    st.markdown("---")
    st.subheader("School performance maps (ICSEA)")

    # Scope SAL shapes to suburb/postcode selection so maps reflect Group 2
    if F.get("suburb"):  # specific suburb chosen
        sal_scope = sal[sal["Suburb"] == F["suburb"]].copy()
    elif F.get("postcode"):  # postcode chosen: keep suburbs present in filtered rows
        sal_scope = sal[sal["Suburb"].isin(filt["Suburb"].dropna().unique())].copy()
    else:
        sal_scope = sal.copy()

    g_icsea = sal_scope.copy()

    if "Primary_School_ICSEA" in filt.columns and not filt.empty:
        prim = (filt.groupby("Suburb", as_index=False)["Primary_School_ICSEA"].mean())
        g_icsea = g_icsea.merge(prim, on="Suburb", how="left")
    else:
        g_icsea["Primary_School_ICSEA"] = np.nan

    if "Secondary_School_ICSEA" in filt.columns and not filt.empty:
        sec = (filt.groupby("Suburb", as_index=False)["Secondary_School_ICSEA"].mean())
        g_icsea = g_icsea.merge(sec, on="Suburb", how="left")
    else:
        g_icsea["Secondary_School_ICSEA"] = np.nan

    prim_cfg, prim_legend = assign_bin_color(g_icsea["Primary_School_ICSEA"])
    sec_cfg,  sec_legend  = assign_bin_color(g_icsea["Secondary_School_ICSEA"])

    g_icsea["prim_color"] = g_icsea["Primary_School_ICSEA"].map(
        lambda v: prim_cfg["colour_fn"](v) if prim_cfg else DEFAULT_GREY
    )
    g_icsea["sec_color"]  = g_icsea["Secondary_School_ICSEA"].map(
        lambda v: sec_cfg["colour_fn"](v) if sec_cfg else DEFAULT_GREY
    )

    vstate = view_for(g_icsea)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Primary (ICSEA)")
        deckA = make_map(
            g_icsea, "prim_color",
            "<b>{Suburb}</b><br/>Primary ICSEA: {Primary_School_ICSEA}", vstate
        )
        st.pydeck_chart(deckA, use_container_width=True)
        render_legend("Legend (Primary ICSEA)", prim_legend)

    with cB:
        st.markdown("#### Secondary (ICSEA)")
        deckB = make_map(
            g_icsea, "sec_color",
            "<b>{Suburb}</b><br/>Secondary ICSEA: {Secondary_School_ICSEA}", vstate
        )
        st.pydeck_chart(deckB, use_container_width=True)
        render_legend("Legend (Secondary ICSEA)", sec_legend)

# ============ TAB 2: GEOGRAPHIC INFO ============
with tab2:
    st.subheader("R zoning & Bushfire")

    filt = apply_all_filters(df)

    df_map = filt.copy()
    if "Top_rcode" not in df_map.columns: df_map["Top_rcode"] = np.nan
    if "pct_overlap" not in df_map.columns: df_map["pct_overlap"] = 0.0
    if "Postcode" not in df_map.columns: df_map["Postcode"] = pd.NA
    if "Suburb" not in df_map.columns: df_map["Suburb"] = pd.NA

    agg = (
        df_map.groupby("Suburb", as_index=False)
              .agg({"Top_rcode": "first", "pct_overlap": "max", "Postcode": "first"})
    )

    gmap = sal.merge(agg, on="Suburb", how="left")

    gmap["rcode_key"] = gmap["Top_rcode"].map(rkey)
    gmap["fill_color"] = gmap["rcode_key"].map(color_for_key)
    gmap["bf_color"] = gmap["pct_overlap"].map(bf_color_from_pct)

    v2 = view_for(gmap)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### R zoning")
        deck = make_map(gmap, "fill_color", "<b>{Suburb}</b><br/>R-code: {Top_rcode}", v2)
        st.pydeck_chart(deck, use_container_width=True)
        keys_present = [k for k in gmap["rcode_key"].dropna().unique().tolist()]
        keys_present.sort(
            key=lambda x: float(re.search(r'R(\d+(?:\.\d+)?)', x).group(1)) if re.search(r'R(\d+(?:\.\d+)?)', x) else 0
        )
        legend_items = [(k, color_for_key(k)) for k in keys_present] or [("No data", DEFAULT_GREY)]
        render_legend("Legend (R zoning)", legend_items)

    with c2:
        st.markdown("#### Bushfire")
        deck2 = make_map(gmap, "bf_color", "<b>{Suburb}</b><br/>Bushfire overlap: {pct_overlap}%", v2)
        st.pydeck_chart(deck2, use_container_width=True)
        bf_legend = [
            ("0%", [230,230,230,160]),
            ("< 5%", [255,229,204,200]),
            ("5–20%", [255,204,153,210]),
            ("20–40%", [255,153,51,220]),
            ("40–60%", [255,127,14,230]),
            ("> 60%", [204,85,0,240]),
        ]
        render_legend("Legend (Bushfire overlap %)", bf_legend)

# ============ TAB 3: PRICING ============
with tab3:
    st.subheader("Median price by suburb (filtered)")
    filt = apply_all_filters(df)

    if {"Suburb","Price"}.issubset(filt.columns) and not filt.empty:
        med = (filt.groupby("Suburb", as_index=False)["Price"].median().rename(columns={"Price":"Median_Price"}))
        gprice = sal.merge(med, on="Suburb", how="left")
        vals = gprice["Median_Price"].dropna()
        if len(vals) >= 4:
            bins = list(np.quantile(vals, [0.25, 0.5, 0.75]))
        elif len(vals) >= 1:
            m = vals.median(); bins = [m*0.9, m, m*1.1]
        else:
            bins = [400000, 700000, 1000000]

        gprice["price_color"] = gprice["Median_Price"].map(lambda v: price_color(v, bins))
        v3 = view_for(gprice)
        deckP = make_map(gprice, "price_color", "<b>{Suburb}</b><br/>Median price: {Median_Price}", v3)
        st.pydeck_chart(deckP, use_container_width=True)

        b1,b2,b3 = [int(x) for x in bins]
        price_legend = [
            (f"< ${b1:,}", [204,229,255,220]),
            (f"${b1:,}–${b2:,}", [153,204,255,220]),
            (f"${b2:,}–${b3:,}", [51,153,255,230]),
            (f"≥ ${b3:,}", [0,102,204,230]),
        ]
        render_legend("Legend (Median Price)", price_legend)
    else:
        st.info("No price data available for the current filters.")

# ============ TAB 4: FINAL SUGGESTION ============
with tab4:
    st.subheader("Final suggestion")
    filt = apply_all_filters(df)

    if filt.empty:
        st.info("No suburbs match your filters yet.")
    else:
        rank_df = filt.copy()
        agg = (
            rank_df.groupby("Suburb", as_index=False)
                   .agg(
                       Median_Price=("Price","median"),
                       Avg_Primary_ICSEA=("Primary_School_ICSEA","mean"),
                       Avg_Secondary_ICSEA=("Secondary_School_ICSEA","mean"),
                       Region=("Region","first"),
                       Postcode=("Postcode","first"),
                       R_Code=("Top_rcode","first"),
                       Distance_km=("Distance_to_CBD_km","mean")
                   )
        )
        agg["Avg_Primary_ICSEA"] = agg["Avg_Primary_ICSEA"].round(0)
        agg["Avg_Secondary_ICSEA"] = agg["Avg_Secondary_ICSEA"].round(0)
        agg["Median_Price"] = agg["Median_Price"].round(0)
        agg["Distance_km"] = agg["Distance_km"].round(1)

        agg_sorted = agg.sort_values(
            by=["Avg_Secondary_ICSEA","Avg_Primary_ICSEA","Median_Price"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        st.markdown("Based on your filtering, the best suburbs for you are:")
        st.dataframe(
            agg_sorted[["Suburb","Postcode","Region","R_Code","Avg_Primary_ICSEA","Avg_Secondary_ICSEA","Median_Price","Distance_km"]],
            use_container_width=True, hide_index=True
        )

# ============ TAB 5: INFO GUIDE (LAST) ============
with tab_info:
    st.subheader("How to read the maps and filters")
    st.markdown("This page gives you a quick guide to the terms used throughout the app.")

    st.markdown("### R–Codes (Residential Density Codes)")
    st.caption("Lower R = larger lots / fewer dwellings. Higher R = smaller lots / more dwellings (e.g., townhouses/apartments).")
    st.markdown("""
- **R10** — Very low density; large lots and typically single detached homes.
- **R12.5** — Very low density; large lots, limited subdivision potential.
- **R15** — Low density; mostly single houses with occasional corner subdivision.
- **R17.5** — Low density; incremental subdivision possible in select cases.
- **R20** — Low density; common suburban single homes, some subdivision potential.
- **R25** — Low–medium density; smaller lots and duplex/villa opportunities.
- **R30** — Medium density; small-lot homes and grouped dwellings become common.
- **R40** — Medium density; **often subdividable** with villas/townhouses feasible.
- **R50** — Medium–higher density; terraces and small apartment formats near centres.
- **R60** — Higher density; townhouses to low-rise apartments around activity nodes.
- **R80** — Higher density; mid-rise apartments around centres/transport corridors.
- **R100** — High density; larger apartment buildings in major centres/precincts.
- **R-AC0** — Activity Centre mixed-use; highest intensity in key precincts.
    """)

    st.markdown("### School ICSEA (Index of Community Socio-Educational Advantage)")
    st.caption("ICSEA has an Australian average of ~1000. It compares the school's student community context; it is **not** a direct measure of teaching quality.")
    st.markdown("""
- **< 900** — Well below average context.
- **900–950** — Below average.
- **950–1050** — Around average (typical band).
- **1050–1100** — Above average context.
- **> 1100** — Well above average context.
    """)
    st.info("Use the ICSEA sliders to set your preferred range; School maps colour suburbs by the **average** ICSEA of schools in that suburb (after your filters).")

    st.markdown("### Bushfire overlap (%)")
    st.caption("Shows the share of a suburb area that intersects designated bushfire-prone vegetation/areas.")
    st.markdown("""
- **0%** — No mapped bushfire-prone overlap.
- **< 5%** — Minor overlap; usually fringe or isolated pockets.
- **5–20%** — Some overlap; localised considerations may apply.
- **20–40%** — Notable overlap; due diligence recommended (BAL, design, insurance).
- **40–60%** — Significant overlap; tighter building/insurance requirements likely.
- **> 60%** — Major overlap; expect stronger constraints (site-specific advice essential).
    """)
    st.warning("Always seek site-specific planning advice (e.g., BAL assessment, local planning scheme) before purchase or development.")
