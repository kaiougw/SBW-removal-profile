# app.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def floatlist(a) -> list:
    return np.asarray(a, dtype=float).ravel().tolist()

def floatlist2d(a) -> list:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return a.tolist()

def _clear_flag(flag_key: str):
    st.session_state[flag_key] = False
    

class sbwinfo(object):
    MachineName = ''
    ProductNo = ''
    RecipeName = ''
    Lot=''
    WaferCount=0
    Recipe={}
    SummaryReport=[]
    WaferData={}
    def __init__(self, Lot=None):
        self.Lot = Lot

# Parsing
def parsesbw(sbwfile: str) -> sbwinfo:
    sbw = sbwinfo()
    if not os.path.exists(sbwfile):
        raise Exception("Invalid sbw File Path!")

    with open(sbwfile, 'r') as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            if line.rstrip() != "":
                if line.rstrip() != '[MeasureData]':
                    raise Exception("Invalid sbw File Format!")
                else:
                    break

        tmpstr = []
        endloop = False
        line = fp.readline()
        while line:
            tmpline = line.replace(' ','').rstrip().split(',')
            if tmpline[0] == '[MeasureData.Header]':
                while line:
                    line = fp.readline()
                    if line.rstrip()!='':
                        tmpstr=line.replace(' ','').rstrip().split(',')
                        if tmpstr[0]=='MachineName':
                            sbw.MachineName=tmpstr[1]
                        elif tmpstr[0]=='ProductNo':
                            sbw.ProductNo=tmpstr[1].rstrip().replace('"','')
                        elif tmpstr[0]=='LotNo':
                            sbw.Lot=tmpstr[1].replace('"','')
                        elif tmpstr[0]=='RecipeName':
                            sbw.RecipeName=tmpstr[1]
                        elif tmpstr[0]=='WaferCount':
                            if tmpstr[1].rstrip().isnumeric():
                                sbw.WaferCount=int(tmpstr[1])
                            break

            elif tmpline[0] == '[MeasureData.WaferDataList]':
                sbw.SummaryReport.clear()
                rptrows=tmpline[1]
                if rptrows.isnumeric():
                    line = fp.readline()
                    rptcol=line.replace(' ','').rstrip().split(',')
                    for _ in range(int(rptrows)):
                        line = fp.readline()
                        tmpstr=line.replace(' ','').rstrip().split(',')
                        _row={}
                        for j in range(1,len(rptcol)):
                            _row[rptcol[j]]=tmpstr[j]
                        sbw.SummaryReport.append(_row)
                        break

            elif tmpline[0] == '[MeasureData.PointsDataList]':
                sbw.WaferData.clear()
                waferno=tmpline[1]
                if waferno.isnumeric():
                    for i in range(int(waferno)):
                        wafer={}
                        while line:
                            line = fp.readline()
                            tmp = line.replace(' ','').rstrip().split(',')
                            if tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}]':
                                line = fp.readline()
                                t1=line.replace(' ','').rstrip().split(',')
                                wafer[t1[0]]=t1[1]
                            elif tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}.AngleDataList]':
                                angle=[]
                                rptrows=tmp[1]
                                if rptrows.isnumeric():
                                    line = fp.readline()
                                    for _ in range(int(rptrows)):
                                        line = fp.readline()
                                        t1=line.replace(' ','').rstrip().split(',')
                                        angle.append(float(t1[1]))
                                    wafer['Angle']=angle
                            elif tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}.LocateList]':
                                radius=[]
                                rptrows=tmp[1]
                                if rptrows.isnumeric():
                                    line = fp.readline()
                                    for _ in range(int(rptrows)):
                                        line = fp.readline()
                                        t1=line.replace(' ','').rstrip().split(',')
                                        radius.append(float(t1[1]))
                                    wafer['Radius']=radius
                            elif tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}.LineDataList]':
                                rptrows=tmp[1]
                                profiles=[]
                                if rptrows.isnumeric():
                                    line = fp.readline()
                                    for j in range(int(rptrows)):
                                        while line:
                                            line = fp.readline()
                                            t1=line.replace(' ','').rstrip().split(',')
                                            if t1[0]==f'[MeasureData.PointsDataList.PointsData_{i}.LineDataList.PointDataList_{j}]':
                                                line = fp.readline()
                                                profile=[]
                                                for _ in range(int(t1[1])):
                                                    line = fp.readline()
                                                    t2=line.replace(' ','').rstrip().split(',')
                                                    profile.append([float(t2[1]),float(t2[2])])
                                                profiles.append(profile)
                                                break
                                    wafer['Profiles']=profiles
                                sbw.WaferData[str(i)]=wafer
                                break
            if endloop:
                break
            line = fp.readline()
    return sbw

def cleansbw(sbwfile) -> Dict[str, Any]:
    lot = getattr(sbwfile, 'Lot', '')
    wd_src = getattr(sbwfile, 'WaferData', {}) or {}
    wd_dst: Dict[str, Any] = {}
    for k, w in wd_src.items():
        is_dict = isinstance(w, dict)
        slotno = (w.get('SlotNo', k) if is_dict else getattr(w, 'SlotNo', k))
        wlot   = (w.get('Lot', lot) if is_dict else getattr(w, 'Lot', lot))
        radius = (w.get('Radius') if is_dict else getattr(w, 'Radius', [])) or []
        angle  = (w.get('Angle')  if is_dict else getattr(w, 'Angle',  [])) or []
        profs  = (w.get('Profiles') if is_dict else getattr(w, 'Profiles', [])) or []
        wd_dst[k] = {
            'SlotNo': slotno,
            'Lot': wlot,
            'Radius': floatlist(radius),
            'Angle': floatlist(angle),
            'Profiles': [floatlist2d(p) for p in profs],
        }
    return {'Lot': lot, 'WaferData': wd_dst}


def gridThk(wafer):
    import numpy as np
    r = np.asarray(wafer.get('Radius', []), dtype=float)
    theta = np.asarray(wafer.get('Angle', []), dtype=float)
    profiles = wafer.get('Profiles', [])
    nt, nr = len(theta), len(r)
    Thk = np.full((nt, nr), np.nan, dtype=float)
    for i in range(nt):
        line = np.asarray(profiles[i], dtype=float) if i < len(profiles) else np.array([], dtype=float)
        if line.ndim == 2 and line.shape[1] > 0:
            Thk[i, :min(nr, line.shape[0])] = line[:nr, 0]
        else:
            Thk[i, :min(nr, line.size)] = line.ravel()[:nr]
    return r, theta, Thk


def gridFlat(wafer):
    import numpy as np
    r = np.asarray(wafer.get('Radius', []), dtype=float)
    theta = np.asarray(wafer.get('Angle', []), dtype=float)
    profiles = wafer.get('Profiles', [])
    nt, nr = len(theta), len(r)
    Flat = np.full((nt, nr), np.nan, dtype=float)
    for i in range(nt):
        line = np.asarray(profiles[i], dtype=float) if i < len(profiles) else np.array([], dtype=float)
        if line.ndim == 2 and line.shape[1] > 1:
            Flat[i, :min(nr, line.shape[0])] = line[:nr, 1]
        else:
            Flat[i, :min(nr, line.size)] = line.ravel()[:nr]
    return r, theta, Flat


@dataclass
class SlotCache:
    r: np.ndarray
    theta: np.ndarray
    Thk: np.ndarray
    Flat: np.ndarray
    line_min_thk: float
    line_max_thk: float
    line_min_flat: float
    line_max_flat: float
    Rmax: float
    X_mir: np.ndarray
    Y_mir: np.ndarray
    Thk_mir: np.ndarray
    Flat_mir: np.ndarray


def _finite_minmax(arr: np.ndarray, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    af = arr[np.isfinite(arr)]
    if af.size == 0:
        return default
    return float(np.min(af)), float(np.max(af))

def _finite_max(arr: np.ndarray, default: float = 0.0) -> float:
    af = arr[np.isfinite(arr)]
    return float(np.max(af)) if af.size else default

def build_slot_cache(wafer_dict) -> SlotCache:
    r, theta, Thk = gridThk(wafer_dict)
    _, _, Flat = gridFlat(wafer_dict)

    lmin_Thk, lmax_Thk = _finite_minmax(Thk, (0.0, 0.0))
    lmin_Flat, lmax_Flat = _finite_minmax(Flat, (0.0, 0.0))
    Rmax = _finite_max(r, 0.0)

    if theta.size and r.size:
        theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi))
        Thk_full = np.vstack([Thk, Thk[:, ::-1]]) if Thk.size else np.empty((0, 0))
        Flat_full = np.vstack([Flat, Flat[:, ::-1]]) if Flat.size else np.empty((0, 0))
        T, Rm = np.meshgrid(theta_full, r, indexing='ij')
        X_mir = Rm*np.cos(T)
        Y_mir = Rm*np.sin(T)
    else:
        Thk_full = np.empty((0, 0))
        Flat_full = np.empty((0, 0))
        X_mir = np.empty((0, 0))
        Y_mir = np.empty((0, 0))

    return SlotCache(
        r=r, theta=theta, Thk=Thk, Flat=Flat,
        line_min_thk=lmin_Thk, line_max_thk=lmax_Thk,
        line_min_flat=lmin_Flat, line_max_flat=lmax_Flat,
        Rmax=Rmax, X_mir=X_mir, Y_mir=Y_mir, Thk_mir=Thk_full, Flat_mir=Flat_full
    )


def graph_arrays(c: SlotCache, graph: str):
    return (c.Flat, c.Flat_mir, 'Flatness (µm)') if graph == 'flat' else (c.Thk, c.Thk_mir, 'Thickness (µm)')


def graph_label(graph: str, prefix: str = "") -> str:
    base = "Flatness" if graph == "flat" else "Thickness"
    if prefix:
        return f"{prefix} {base}"
    return f"{base}"


@st.cache_data(show_spinner=False)
def parse_and_clean(uploaded_bytes: bytes) -> Dict[str, Any]:
    import tempfile
    obj = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sbw") as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = tmp.name
    try:
        obj = parsesbw(tmp_path)
        return cleansbw(obj)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@st.cache_data(show_spinner=False)
def cache_for_data(data: Dict[str, Any]) -> Dict[str, SlotCache]:
    wafers = data.get('WaferData', {}) or {}
    return {k: build_slot_cache(w) for k, w in wafers.items()}


def robust_clip(Z: np.ndarray, p_lo: float, p_hi: float):
    Zf = Z[np.isfinite(Z)]
    if Zf.size == 0:
        return Z, 0.0, 1.0
    vmin = float(np.nanpercentile(Zf, p_lo))
    vmax = float(np.nanpercentile(Zf, p_hi))
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = vmin + 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-9
    return np.clip(Z, vmin, vmax), vmin, vmax


def mask_outliers(Z: np.ndarray, k: float=4): # Outlier threshold = 4
    Zm = Z.astype(float, copy=True)
    Zf = Zm[np.isfinite(Zm)]
    if Zf.size == 0:
        return Zm
    med = float(np.nanmedian(Zf))
    mad = float(np.nanmedian(np.abs(Zf - med))) * 1.4826 
    if mad == 0 or not np.isfinite(mad):
        return Zm
    Zm[np.abs(Zm - med) > k * mad] = np.nan # Distance > k x MAD is marked as outlier
    return Zm


def plot_3d(X, Y, Z, zlabel: str, p_lo: float, p_hi: float, do_mask: bool, height: int = 600):
    if Z.size == 0:
        return
    Zg = mask_outliers(Z) if do_mask else Z
    Zc, vmin, vmax = robust_clip(Zg, p_lo, p_hi)
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Zg,
            surfacecolor=Zc, colorscale="Jet",
            cmin=vmin, cmax=vmax,
            colorbar=dict(title=zlabel, len=0.8, thickness=15),
            contours = {
            "z": {
                "show": True,
                "usecolormap": True,
                "highlight": True,
                "project": {"z": True},
                "start": vmin,
                "end": vmax,
                "size": (vmax-vmin)/20
            }
        }
        )
    ])
    fig.update_scenes(
        xaxis_title="Radius (mm)",
        yaxis_title="Radius (mm)",
        zaxis_title=zlabel,
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.7)
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, autosize=True,)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def plot_2d(X, Y, Z, zlabel: str, radius_max: float, p_lo: float, p_hi: float, do_mask: bool, height: int=600):
    if Z.size == 0:
        return
    Zg = mask_outliers(Z) if do_mask else Z
    Zc, vmin, vmax = robust_clip(Zg, p_lo, p_hi)
    z0 = np.zeros_like(Zc)
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=z0,
            surfacecolor=Zc, colorscale="Jet",
            cmin=vmin, cmax=vmax, showscale=True,
            colorbar=dict(title=zlabel, len=0.8, thickness=15)
        )
    ])
    theta = np.linspace(0, 2*np.pi, 200)
    rmax = radius_max if np.isfinite(radius_max) and radius_max > 0 else 0.0
    cx, cy = rmax * np.cos(theta), rmax * np.sin(theta)
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=[0]*len(cx),
        mode="lines", line=dict(color="black", width=2),
        showlegend=False
    ))
    fig.update_scenes(
        zaxis=dict(visible=False),
        xaxis_title="Radius (mm)",
        yaxis_title="Radius (mm)",
        aspectmode="data",
        camera=dict(eye=dict(x=0, y=0, z=15), up=dict(x=-1, y=0, z=0))
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),dragmode="pan", height=height, autosize=True,)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

def _avg_lines(cache_dict: Dict[str, SlotCache], slots: List[str], graph: str):
    lines = []
    rs = []
    thetas = []
    for s in slots:
        c = cache_dict.get(s)
        if not c:
            continue
        L, _, _ = graph_arrays(c, graph)
        if L.size == 0:
            continue
        lines.append(L)
        rs.append(c.r)
        thetas.append(c.theta)
    if not lines:
        return None
    nt = min(L.shape[0] for L in lines)
    nr = min(L.shape[1] for L in lines)
    lines = [L[:nt, :nr] for L in lines]
    r = rs[0][:nr] if rs else np.array([])
    theta = thetas[0][:nt] if thetas else np.array([])
    mean = np.nanmean(np.stack(lines, axis=0), axis=0)
    return mean, r, theta

def finite_xy(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    return (x[m], y[m]) if m.any() else (np.array([]), np.array([]))


def plot_line_profile(r: np.ndarray, line: np.ndarray, zlabel: str, title: str, height: int = 500,
                      overlay_pre: Optional[np.ndarray] = None, overlay_post: Optional[np.ndarray] = None):
    x = np.asarray(r, dtype=float)
    y = np.asarray(line, dtype=float)
    x, y = finite_xy(-x, y) # -x flips lines horizontally

    fig = go.Figure()

    if overlay_pre is not None:
        _, y_pre = finite_xy(x, np.asarray(overlay_pre, dtype=float))
        fig.add_trace(go.Scatter(
            x=x[:y_pre.size], y=y_pre, mode="lines", 
            name="PRE", line=dict(width=1.0, color="lightgray")
        ))
    if overlay_post is not None:
        _, y_post = finite_xy(x, np.asarray(overlay_post, dtype=float))
        fig.add_trace(go.Scatter(
            x=x[:y_post.size], y=y_post, mode="lines",
            name="POST", line=dict(width=1.0, color="lightgray")
        ))

    if y.size:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="red"),
            name="Removal"
        ))

    fig.update_layout(
        margin=dict(l=30, r=30, t=10, b=30),
        xaxis_title="Radius (mm)",
        yaxis_title=zlabel,
        hovermode="x unified",
        dragmode="pan",
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False)
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def plot_line_grid(r: np.ndarray, theta: np.ndarray, Z_line: np.ndarray, zlabel: str,
                   nrows=2, ncols=4, height: int = 650,
                   overlay_pre: Optional[np.ndarray] = None, overlay_post: Optional[np.ndarray] = None):
    r = np.asarray(r, dtype=float)
    if Z_line.size == 0:
        return
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=True)
    n = Z_line.shape[0]
    count = min(n, nrows * ncols)

    for i in range(count):
        row, col = i // ncols + 1, i % ncols + 1
        y = np.asarray(Z_line[i, :], dtype=float)
        x_i, y_i = finite_xy(-r, y) # -r flips lines horizontally
        ang = np.degrees(theta[i]) if i < len(theta) and np.isfinite(theta[i]) else np.nan
        label = f"Angle {ang:.1f}°"

        if overlay_pre is not None and overlay_pre.size:
            y_pre = np.asarray(overlay_pre[i, :], dtype=float) if i < overlay_pre.shape[0] else np.array([], dtype=float)
            x_pre, y_pre = finite_xy(r, y_pre)
            fig.add_trace(
                go.Scatter(x=x_pre, y=y_pre, mode="lines",
                           line=dict(width=1.0, color="lightgray"),
                           name="PRE"),
                row=row, col=col
            )
        if overlay_post is not None and overlay_post.size:
            y_post = np.asarray(overlay_post[i, :], dtype=float) if i < overlay_post.shape[0] else np.array([], dtype=float)
            x_post, y_post = finite_xy(r, y_post)
            fig.add_trace(
                go.Scatter(x=x_post, y=y_post, mode="lines",
                           line=dict(width=1.0, color="lightgray"),
                           name="POST"),
                row=row, col=col
            )

        fig.add_trace(
            go.Scatter(
                x=x_i, y=y_i, mode="lines",
                name="Angle",
                line=dict(width=1.2, color="red"),
                showlegend=False,
                hovertemplate="x: %{x}<br>y: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
        label = f"Angle {ang+180:.1f}°"
        fig.add_annotation(
            text=label,
            showarrow=False,
            x=0.5, xref="x domain",
            y=1.15, yref="y domain",
            font=dict(size=12, color="gray"),
            row=row, col=col
        )

        if row == nrows:
            fig.update_xaxes(title_text="Radius (mm)", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=zlabel, row=row, col=col)

    fig.update_layout(showlegend=False, dragmode="pan", height=height)
    for r_i in range(1, nrows+1):
        for c_i in range(1, ncols+1):
            fig.update_xaxes(matches="x1", row=r_i, col=c_i, showticklabels=True)
            fig.update_yaxes(matches="y1", row=r_i, col=c_i, showticklabels=True)
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def sorted_keys(d):
    try:
        return sorted(d.keys(), key=lambda k: float(k))
    except Exception:
        return sorted(d.keys(), key=str)


def slot_options(data: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not data:
        return []
    disp = []
    wafers = data.get('WaferData', {}) or {}
    for k in sorted_keys(wafers):
        ref = wafers.get(k, {}) or {}
        disp.append((f"Slot {ref.get('SlotNo', k)}", k))
    return disp

# UI
st.set_page_config(page_title="SBW Removal Profile", layout="wide")
st.title("SBW Removal Profile")

with st.sidebar:
    st.markdown("**Display controls**")
    p_lo = st.slider("Color clip low (%)", 0.0, 5.0, 0.5, 0.5) # To adjust lowest percentile value
    p_hi = st.slider("Color clip high (%)", 95.0, 100.0, 100.0, 1.0) # To adjust highest percentile value
    if p_hi <= p_lo:
        p_hi = min(100.0, p_lo + 0.5)
    do_mask = st.checkbox("Mask notch", value=False)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    pre_file = st.file_uploader("Upload PRE .sbw", type=["sbw"], key="pre")
with colB:
    post_file = st.file_uploader("Upload POST .sbw", type=["sbw"], key="post")
with colC:
    graph = st.selectbox(
        "",
        options=[("Thickness", "thk"), ("Flatness", "flat")],
        format_func=lambda x: x[0]
    )[1]
    profile_mode = st.segmented_control("",["PRE", "POST", "REMOVAL"],label_visibility="hidden", width="stretch")

# Sidebar options only when REMOVAL is selected
# show_prepost_3d = False
overlay_prepost_lines = False
if profile_mode == "REMOVAL":
    with st.sidebar:
        # show_prepost_3d = st.checkbox("PRE/POST 3D plots", value=False)
        overlay_prepost_lines = st.checkbox("Overlay line charts", value=False)

PRE_DATA = POST_DATA = None
PRE_CACHE = POST_CACHE = None

if pre_file is not None:
    try:
        PRE_DATA = parse_and_clean(pre_file.read())
        PRE_CACHE = cache_for_data(PRE_DATA)
        st.success(f"Loaded {PRE_DATA.get('Lot', '')}")
    except Exception as e:
        st.error(f"Failed to parse PRE: {e}")

if post_file is not None:
    try:
        POST_DATA = parse_and_clean(post_file.read())
        POST_CACHE = cache_for_data(POST_DATA)
        st.success(f"Loaded {POST_DATA.get('Lot', '')}")
    except Exception as e:
        st.error(f"Failed to parse POST: {e}")

st.markdown("---")

if profile_mode in ("PRE", "POST"):
    data = PRE_DATA if profile_mode == "PRE" else POST_DATA
    cache = PRE_CACHE if profile_mode == "PRE" else POST_CACHE
    if not data or not cache:
        st.info(f"Please upload a {profile_mode} file.")
    else:
        opts = slot_options(data)
        labels = [label for label, _ in opts]
        values = [val for _, val in opts]
        plot_key = f"do_plot_{profile_mode}"
        sel = st.multiselect("Slots", labels, default=None, key=f"{profile_mode}_slots",
                            on_change=_clear_flag, args=(plot_key,))

        sel_keys = [values[labels.index(lbl)] for lbl in sel] if sel else []
        if st.button("Plot", key=f"plot_btn_{profile_mode}"):
            st.session_state[plot_key] = True
        if st.session_state.get(plot_key, False):
            if not sel_keys:
                st.warning("Choose at least one slot.")
            for slot in sel_keys:
                if slot not in cache:
                    st.warning(f"No cache for slot {slot}")
                    continue
                c = cache[slot]
                r, theta = c.r, c.theta
                Z_line, Z_surf, zlabel = graph_arrays(c, graph)
                if Z_line.size == 0 or Z_surf.size == 0:
                    st.warning(f"No data in slot {slot}")
                    continue
                X, Y = c.X_mir, c.Y_mir
                lot = data.get('WaferData', {}).get(slot, {}).get('Lot', data.get('Lot', ''))
                slotno = data.get('WaferData', {}).get(slot, {}).get('SlotNo', slot)

                st.subheader(f"{graph_label(graph)}\n{lot}({slotno})")

                col1, col2 = st.columns(2)
                with col1:
                    plot_2d(X, Y, Z_surf, zlabel, c.Rmax, p_lo, p_hi, do_mask)
                with col2:
                    plot_3d(X, Y, Z_surf, zlabel, p_lo, p_hi, do_mask)

                plot_line_grid(r, theta, Z_line, zlabel, nrows=2, ncols=4, height=650)

                if len(theta) > 0:
                    angle_options = [f"{np.degrees(a)+180:.1f}°" for a in theta]
                    ang_key = f"ang_{profile_mode}_{slot}"
                    if ang_key not in st.session_state:
                        st.session_state[ang_key] = angle_options[0]
                    ang_str = st.select_slider("Angle", options=angle_options, key=ang_key)
                    idx = angle_options.index(ang_str)
                    ang = theta[idx]
                    line = Z_line[idx, :]
                    plot_line_profile(r, line, zlabel, f"Angle {ang+180:.1f}°", height=520)


                st.markdown("---")

else:
    if not (PRE_DATA and POST_DATA and PRE_CACHE and POST_CACHE):
        st.info("Please upload both PRE and POST files.")
    else:
        pre_opts = slot_options(PRE_DATA)
        post_opts = slot_options(POST_DATA)
        pre_labels = [l for l, _ in pre_opts]
        pre_values = [v for _, v in pre_opts]
        post_labels = [l for l, _ in post_opts]
        post_values = [v for _, v in post_opts]

        plot_key = "do_plot_REMOVAL"

        c1, c2 = st.columns(2)
        with c1:
            sel_pre = st.multiselect(
                "PRE slots", pre_labels, default=None,
                key="rem_pre_slots", on_change=_clear_flag, args=(plot_key,)
            )
            pre_keys = [pre_values[pre_labels.index(lbl)] for lbl in sel_pre] if sel_pre else []
        with c2:
            sel_post = st.multiselect(
                "POST slots", post_labels, default=None,
                key="rem_post_slots", on_change=_clear_flag, args=(plot_key,)
            )
            post_keys = [post_values[post_labels.index(lbl)] for lbl in sel_post] if sel_post else []

        if st.button("Plot", key="plot_btn_REMOVAL"):
            st.session_state[plot_key] = True

        if st.session_state.get(plot_key, False):
            if not pre_keys or not post_keys:
                st.warning("Choose at least one PRE slot and one POST slot.")
            n_pairs = min(len(pre_keys), len(post_keys))
            if len(pre_keys) != len(post_keys) and n_pairs > 0:
                st.info(f"Pairing first {n_pairs} slots in order.")
            for pre_slot, post_slot in zip(pre_keys[:n_pairs], post_keys[:n_pairs]):
                if pre_slot not in PRE_CACHE or post_slot not in POST_CACHE:
                    st.warning("Selected slot missing in cache.")
                    continue
                A_c, B_c = PRE_CACHE[pre_slot], POST_CACHE[post_slot]
                r, theta = A_c.r, A_c.theta
                A_line, A_surf, _ = graph_arrays(A_c, graph)
                B_line, B_surf, _ = graph_arrays(B_c, graph)
                if A_line.size == 0 or B_line.size == 0:
                    st.warning("No overlapping data for removal.")
                    continue
                nt = min(A_line.shape[0], B_line.shape[0])
                nr = min(A_line.shape[1], B_line.shape[1])
                if nt == 0 or nr == 0:
                    st.warning("No overlapping data for removal.")
                    continue
                r = r[:nr]
                theta = theta[:nt]
                Z_line = B_line[:nt, :nr] - A_line[:nt, :nr]
                Z_surf = np.vstack([Z_line, Z_line[:, ::-1]])
                theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi))
                T, Rm = np.meshgrid(theta_full, r, indexing='ij')
                X = Rm * np.cos(T)
                Y = Rm * np.sin(T)
                zlabel = 'Removal (µm)'

                pre_lot = PRE_DATA.get('WaferData', {}).get(pre_slot, {}).get('Lot', PRE_DATA.get('Lot', ''))
                post_lot = POST_DATA.get('WaferData', {}).get(post_slot, {}).get('Lot', POST_DATA.get('Lot', ''))
                pre_slotno = PRE_DATA.get('WaferData', {}).get(pre_slot, {}).get('SlotNo', pre_slot)
                post_slotno = POST_DATA.get('WaferData', {}).get(post_slot, {}).get('SlotNo', post_slot)

                st.subheader(f"{graph_label(graph)} Removal Profile\n{pre_lot}({pre_slotno}), {post_lot}({post_slotno})")

                c1, c2 = st.columns(2)
                with c1:
                    rmax = float(np.max(r[np.isfinite(r)])) if np.isfinite(r).any() else 0.0
                    plot_2d(X, Y, Z_surf, zlabel, rmax, p_lo, p_hi, do_mask)
                with c2:
                    # plot_3d(A_c.X_mir, A_c.Y_mir, A_surf, graph_label(graph, "PRE"), p_lo, p_hi, do_mask, height=300)
                    # plot_3d(B_c.X_mir, B_c.Y_mir, B_surf, graph_label(graph, "POST"), p_lo, p_hi, do_mask, height=300)

                    view_key = f"show3d_{pre_slot}_{post_slot}"
                    if view_key not in st.session_state:
                        st.session_state[view_key] = False

                    label = "◀" if st.session_state[view_key] else "▶"

                    if st.button(label, key=f"btn_{pre_slot}_{post_slot}"):
                        st.session_state[view_key] = not st.session_state[view_key]
                        st.rerun() 

                    if st.session_state[view_key]:
                        plot_3d(A_c.X_mir, A_c.Y_mir, A_surf, graph_label(graph, "PRE"), p_lo, p_hi, do_mask, height=300)
                        plot_3d(B_c.X_mir, B_c.Y_mir, B_surf, graph_label(graph, "POST"), p_lo, p_hi, do_mask, height=300)
                    else:
                        plot_2d(A_c.X_mir, A_c.Y_mir, A_surf, graph_label(graph, "PRE"), A_c.Rmax, p_lo, p_hi, do_mask, height=300)
                        plot_2d(B_c.X_mir, B_c.Y_mir, B_surf, graph_label(graph, "POST"), B_c.Rmax, p_lo, p_hi, do_mask, height=300)

                overlay_pre = A_line[:nt, :nr] if overlay_prepost_lines else None
                overlay_post = B_line[:nt, :nr] if overlay_prepost_lines else None

                plot_line_grid(r, theta, Z_line, zlabel, nrows=2, ncols=4, height=650,
                               overlay_pre=overlay_pre, overlay_post=overlay_post)

                if len(theta) > 0:
                    angle_options = [f"{np.degrees(a)+180:.1f}°" for a in theta]
                    ang_key = f"ang_rem_{pre_slot}_{post_slot}"
                    if ang_key not in st.session_state:
                        st.session_state[ang_key] = angle_options[0]
                    ang_str = st.select_slider("Angle", options=angle_options, key=ang_key)
                    idx = angle_options.index(ang_str)
                    ang = theta[idx]
                    line = Z_line[idx, :]
                    pre_overlay_line = overlay_pre[idx, :] if overlay_pre is not None else None
                    post_overlay_line = overlay_post[idx, :] if overlay_post is not None else None
                    plot_line_profile(
                        r, line, zlabel, f"Angle {ang+180:.1f}°",
                        height=520,
                        overlay_pre=pre_overlay_line,
                        overlay_post=post_overlay_line
                    )

                st.markdown("---")
