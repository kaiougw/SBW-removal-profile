app.py
from __future__ import annotations
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm


def safe_get(d: dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def floatlist(a) -> list:
    return np.asarray(a, dtype=float).ravel().tolist()

def floatlist2d(a) -> list:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return a.tolist()


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
                    line = fp.readline() # column name row
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
                            line = fp.readline() # column name
                            tmp = line.replace(' ','').rstrip().split(',')
                            if tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}]':
                                line = fp.readline() # next line: SlotNo
                                t1=line.replace(' ','').rstrip().split(',')
                                wafer[t1[0]]=t1[1]  # SlotNo
                            elif tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}.AngleDataList]':
                                angle=[]
                                rptrows=tmp[1]
                                if rptrows.isnumeric():
                                    line = fp.readline() # skip column header
                                    for _ in range(int(rptrows)):
                                        line = fp.readline()
                                        t1=line.replace(' ','').rstrip().split(',')
                                        angle.append(float(t1[1]))
                                    wafer['Angle']=angle
                            elif tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}.LocateList]':
                                radius=[]
                                rptrows=tmp[1]
                                if rptrows.isnumeric():
                                    line = fp.readline() # skip column header
                                    for _ in range(int(rptrows)):
                                        line = fp.readline()
                                        t1=line.replace(' ','').rstrip().split(',')
                                        radius.append(float(t1[1]))
                                    wafer['Radius']=radius
                            elif tmp[0]==f'[MeasureData.PointsDataList.PointsData_{i}.LineDataList]':
                                rptrows=tmp[1]
                                profiles=[]
                                if rptrows.isnumeric():
                                    line = fp.readline() # skip column header
                                    for j in range(int(rptrows)):
                                        while line:
                                            line = fp.readline()
                                            t1=line.replace(' ','').rstrip().split(',')
                                            if t1[0]==f'[MeasureData.PointsDataList.PointsData_{i}.LineDataList.PointDataList_{j}]':
                                                line = fp.readline() # skip col header
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
        radius = (w.get('Radius') if is_dict else getattr(w, 'Radius', []))
        angle  = (w.get('Angle')  if is_dict else getattr(w, 'Angle',  []))
        profs  = (w.get('Profiles') if is_dict else getattr(w, 'Profiles', []))
        wd_dst[k] = {
            'SlotNo': slotno,
            'Lot': wlot,
            'Radius': floatlist(radius),
            'Angle': floatlist(angle),
            'Profiles': [floatlist2d(p) for p in profs],
        }
    return {'Lot': lot, 'WaferData': wd_dst}

def gridThk(wafer):
    r = np.asarray(wafer['Radius'], dtype=float)
    theta = np.asarray(wafer['Angle'], dtype=float)
    profiles = wafer['Profiles']
    nt, nr = len(theta), len(r)
    Thk = np.full((nt, nr), np.nan, dtype=float)
    for i in range(nt):
        line = np.asarray(profiles[i], dtype=float)
        if line.ndim == 2 and line.shape[1] > 0:
            Thk[i, :min(nr, line.shape[0])] = line[:nr, 0]
        else:
            Thk[i, :min(nr, line.size)] = line.ravel()[:nr]
    return r, theta, Thk

def gridFlat(wafer):
    r = np.asarray(wafer['Radius'], dtype=float)
    theta = np.asarray(wafer['Angle'], dtype=float)
    profiles = wafer['Profiles']
    nt, nr = len(theta), len(r)
    Flat = np.full((nt, nr), np.nan, dtype=float)
    for i in range(nt):
        line = np.asarray(profiles[i], dtype=float)
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

def build_slot_cache(wafer_dict) -> SlotCache:
    r, theta, Thk = gridThk(wafer_dict)
    _, _, Flat = gridFlat(wafer_dict)
    lmin_Thk = float(np.nanmin(Thk)) if Thk.size else 0.0
    lmax_Thk = float(np.nanmax(Thk)) if Thk.size else 0.0
    lmin_Flat = float(np.nanmin(Flat)) if Flat.size else 0.0
    lmax_Flat = float(np.nanmax(Flat)) if Flat.size else 0.0
    Rmax = float(np.nanmax(r)) if r.size else 0.0
    if theta.size and r.size:
        theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi))
        Thk_full = np.vstack([Thk, Thk[:, ::-1]])
        Flat_full = np.vstack([Flat, Flat[:, ::-1]])
        T, Rm = np.meshgrid(theta_full, r, indexing='ij')
        X_mir = Rm*np.cos(T); Y_mir = Rm*np.sin(T)
    else:
        X_mir = Y_mir = Thk_full = Flat_full = np.array([])
    return SlotCache(
        r=r, theta=theta, Thk=Thk, Flat=Flat,
        line_min_thk=lmin_Thk, line_max_thk=lmax_Thk,
        line_min_flat=lmin_Flat, line_max_flat=lmax_Flat,
        Rmax=Rmax, X_mir=X_mir, Y_mir=Y_mir, Thk_mir=Thk_full, Flat_mir=Flat_full
    )

def graph_arrays(c: SlotCache, graph: str):
    return (c.Flat, c.Flat_mir, 'Flatness (µm)') if graph == 'flat' else (c.Thk, c.Thk_mir, 'Thickness (µm)')

def graph_label(graph: str):
    return 'Flatness' if graph == 'flat' else 'Thickness'


@st.cache_data(show_spinner=False)
def parse_and_clean(uploaded_bytes: bytes) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sbw") as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = tmp.name
    try:
        obj = parsesbw(tmp_path)
        return cleansbw(obj)
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

@st.cache_data(show_spinner=False)
def cache_for_data(data: Dict[str, Any]) -> Dict[str, SlotCache]:
    return {k: build_slot_cache(w) for k, w in data['WaferData'].items()}


def plot_surface_3d(X, Y, Z, zlabel: str):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_xlabel('Radius (mm)', fontsize=8)
    ax.set_ylabel('Radius (mm)', fontsize=8)
    ax.set_zlabel(zlabel, fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_map_2d(X, Y, Z, zlabel: str, radius_max: float):
    fig, ax = plt.subplots(figsize=(5, 4))
    # Use pcolor to avoid the pcolormesh monotonic warning
    pc = ax.pcolor(X, Y, Z, cmap=cm.jet)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8); cb.set_label(zlabel, fontsize=8)
    cb.ax.tick_params(labelsize=8)
    circ = plt.Circle((0, 0), radius_max, fill=False, linewidth=1.0, color='k')
    ax.add_artist(circ)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Radius (mm)', fontsize=8)
    ax.set_ylabel('Radius (mm)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_line_profile(r: np.ndarray, line: np.ndarray, zlabel: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(r, line, linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel(zlabel)
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis='both', labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_line_grid(r: np.ndarray, theta: np.ndarray, Z_line: np.ndarray, zlabel: str, nrows=2, ncols=4):
    n = Z_line.shape[0]
    count = min(n, nrows*ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5), sharex=True, sharey=True)
    axes = np.array(axes).ravel()
    for i in range(nrows*ncols):
        ax = axes[i]
        if i < count:
            ax.plot(r, Z_line[i, :], linewidth=0.8)
            ang = theta[i] if i < len(theta) else np.nan
            ax.set_title(f"Angle {ang:.2f}", fontsize=8)
            ax.grid(True, alpha=0.35)
            if i >= ncols: ax.set_xlabel('Radius (mm)', fontsize=8)
            if i % ncols == 0: ax.set_ylabel(zlabel, fontsize=8)
        else:
            ax.axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close(fig)


def sorted_keys(d):
    try:    return sorted(d.keys(), key=lambda k: float(k))
    except: return sorted(d.keys(), key=str)

def slot_options(data: Optional[Dict[str, Any]]) -> List[Tuple[str,str]]:
    if not data: return []
    disp = []
    for k in sorted_keys(data['WaferData']):
        ref = data['WaferData'].get(k, {})
        disp.append((f"Slot {ref.get('SlotNo', k)}", k))
    return disp


st.set_page_config(page_title="SBW Wafer Dashboard", layout="wide")
st.title("SBW Wafer Dashboard")

colA, colB, colC = st.columns([1,1,1])

with colA:
    pre_file = st.file_uploader("Upload PRE .sbw", type=["sbw"], key="pre")
with colB:
    post_file = st.file_uploader("Upload POST .sbw", type=["sbw"], key="post")
with colC:
    graph = st.selectbox("Metric", options=[("Thickness", "thk"), ("Flatness", "flat")], format_func=lambda x:x[0])[1]
    profile_mode = st.radio("Profile", ["PRE", "POST", "REMOVAL"], horizontal=True)

# Parse / cache data
PRE_DATA = POST_DATA = None
PRE_CACHE = POST_CACHE = None

if pre_file is not None:
    try:
        PRE_DATA = parse_and_clean(pre_file.read())
        PRE_CACHE = cache_for_data(PRE_DATA)
        st.success(f"Loaded PRE: {PRE_DATA.get('Lot','')}")
    except Exception as e:
        st.error(f"Failed to parse PRE: {e}")

if post_file is not None:
    try:
        POST_DATA = parse_and_clean(post_file.read())
        POST_CACHE = cache_for_data(POST_DATA)
        st.success(f"Loaded POST: {POST_DATA.get('Lot','')}")
    except Exception as e:
        st.error(f"Failed to parse POST: {e}")

st.markdown("---")

if profile_mode in ("PRE", "POST"):
    data = PRE_DATA if profile_mode == "PRE" else POST_DATA
    cache = PRE_CACHE if profile_mode == "PRE" else POST_CACHE
    if not data:
        st.info(f"Please upload a {profile_mode} file.")
    else:
        opts = slot_options(data)
        labels = [label for label,_ in opts]
        values = [val for _,val in opts]
        sel = st.multiselect(f"{profile_mode} slots", labels, default=labels, key=f"{profile_mode}_slots")
        sel_keys = [values[labels.index(lbl)] for lbl in sel] if sel else []
        if st.button("Plot"):
            if not sel_keys:
                st.warning("Choose at least one slot.")
            for slot in sel_keys:
                c = cache[slot]
                r, theta = c.r, c.theta
                Z_line, Z_surf, zlabel = graph_arrays(c, graph)
                if not Z_line.size:
                    st.warning(f"No data in slot {slot}")
                    continue
                X, Y = c.X_mir, c.Y_mir
                lot = data['WaferData'][slot].get('Lot', data.get('Lot', ''))
                slotno = data['WaferData'][slot].get('SlotNo', slot)

                st.subheader(f"{lot}({slotno}) — {graph_label(graph)}")

                c1, c2 = st.columns(2)
                with c1:
                    st.caption("3D Surface")
                    plot_surface_3d(X, Y, Z_surf, zlabel)
                with c2:
                    st.caption("2D Map")
                    plot_map_2d(X, Y, Z_surf, zlabel, c.Rmax)

                st.caption("Profiles")
                # Grid of first 8 line profiles
                plot_line_grid(r, theta, Z_line, zlabel, nrows=2, ncols=4)

                # Single line viewer
                if len(theta) > 0:
                    idx = st.slider("Angle index", 0, len(theta)-1, 0, key=f"idx_{profile_mode}_{slot}")
                    line = Z_line[idx, :]
                    ang = theta[idx]
                    plot_line_profile(r, line, zlabel, f"Angle {ang:.2f}")
                st.markdown("---")

else:  # REMOVAL
    if not (PRE_DATA and POST_DATA):
        st.info("Please upload both PRE and POST files.")
    else:
        pre_opts = slot_options(PRE_DATA); post_opts = slot_options(POST_DATA)
        pre_labels = [l for l,_ in pre_opts]; pre_values = [v for _,v in pre_opts]
        post_labels = [l for l,_ in post_opts]; post_values = [v for _,v in post_opts]

        c1, c2 = st.columns(2)
        with c1:
            sel_pre = st.multiselect("PRE slots", pre_labels, default=pre_labels)
            pre_keys = [pre_values[pre_labels.index(lbl)] for lbl in sel_pre] if sel_pre else []
        with c2:
            sel_post = st.multiselect("POST slots", post_labels, default=post_labels)
            post_keys = [post_values[post_labels.index(lbl)] for lbl in sel_post] if sel_post else []

        if st.button("Plot Removal"):
            if not pre_keys or not post_keys:
                st.warning("Choose at least one PRE slot and one POST slot.")
            n_pairs = min(len(pre_keys), len(post_keys))
            if len(pre_keys) != len(post_keys) and n_pairs > 0:
                st.info(f"Pairing first {n_pairs} slots in order.")
            for pre_slot, post_slot in zip(pre_keys[:n_pairs], post_keys[:n_pairs]):
                A_c, B_c = PRE_CACHE[pre_slot], POST_CACHE[post_slot]
                r, theta = A_c.r, A_c.theta
                A_line, _, _ = graph_arrays(A_c, graph)
                B_line, _, _ = graph_arrays(B_c, graph)
                nt = min(A_line.shape[0], B_line.shape[0]); nr = min(A_line.shape[1], B_line.shape[1])
                if nt == 0 or nr == 0:
                    st.warning("No overlapping data for removal."); continue
                r = r[:nr]; theta = theta[:nt]
                Z_line = B_line[:nt, :nr] - A_line[:nt, :nr]
                Z_surf = np.vstack([Z_line, Z_line[:, ::-1]])
                theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi))
                T, Rm = np.meshgrid(theta_full, r, indexing='ij')
                X = Rm*np.cos(T); Y = Rm*np.sin(T)
                zlabel = 'Removal (µm)'

                pre_lot  = PRE_DATA['WaferData'][pre_slot].get('Lot',  PRE_DATA.get('Lot', ''))
                post_lot = POST_DATA['WaferData'][post_slot].get('Lot', POST_DATA.get('Lot', ''))
                pre_slotno  = PRE_DATA['WaferData'][pre_slot].get('SlotNo', pre_slot)
                post_slotno = POST_DATA['WaferData'][post_slot].get('SlotNo', post_slot)

                st.subheader(f"{pre_lot}({pre_slotno}), {post_lot}({post_slotno}) — {graph_label(graph)} Removal")

                c3, c4 = st.columns(2)
                with c3:
                    st.caption("3D Surface (Removal)")
                    plot_surface_3d(X, Y, Z_surf, zlabel)
                with c4:
                    st.caption("2D Map (Removal)")
                    plot_map_2d(X, Y, Z_surf, zlabel, float(np.max(r)))

                st.caption("Removal Profiles")
                plot_line_grid(r, theta, Z_line, zlabel, nrows=2, ncols=4)
                if len(theta) > 0:
                    idx = st.slider("Angle index (Removal)", 0, len(theta)-1, 0, key=f"idx_rem_{pre_slot}_{post_slot}")
                    line = Z_line[idx, :]
                    ang = theta[idx]
                    plot_line_profile(r, line, zlabel, f"Angle {ang:.2f} (Removal)")
                st.markdown("---")
