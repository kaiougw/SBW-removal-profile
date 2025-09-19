# app.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
import requests
from io import BytesIO


# Utility Functions
def floatlist(a) -> list:
    """
    Convert array-like input into flat list of floats to ensure numbers are in numeric, not str or int.
    Called inside `cleansbw` for Radius and Angle.
    """
    return np.asarray(a, dtype=float).ravel().tolist()

def floatlist2d(a) -> list:
    """
    Convert array-like input into 2D list of floats to ensure numbers are in numeric, not str or int.
    Called inside `cleansbw` for Profile to preserve its 2D structure ([thickness, flatness] for each radius).
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None] # convert shape (N,) → (N,1)
    return a.tolist()

def reset_plot(flag_key: str):
    """
    A reset switch to control session state.
    Changing slot options resets session state, requiring Plot button to be pressed again.
    """
    st.session_state[flag_key] = False # False -> no plotting until Plot button is pressed

def sort_keys(d): # d = WaferData
    """
    Sorts dictionary keys (slot IDs) as floats if possible, otherwise strings
    """
    try:
        return sorted(d.keys(), key=lambda k: float(k))
    except Exception:
        return sorted(d.keys(), key=str)

def finite_xy(x: np.ndarray, y: np.ndarray):
    """
    Filter out NaN/Inf values from (x, y) pairs.
    Used in `plot_line_profile` and `plot_line_grid` to ensure line plots ignore invalid data points.

    Input
    ---
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.

    Ouput
    ----
    tuple of np.ndarray
        (x_filtered, y_filtered) as arrays
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        return (np.array([]), np.array([]))
    return (x[m], y[m])

def average_profile(Z_line: np.ndarray) -> np.ndarray:
    """
    Compute average radial profile by combining both +r and -r sides

    Input
    ---
    Z_line: np.ndarray
        2D array, shape (n_lines, n_radii), where each row is a scan line along radius positions.

    Output
    ---
    np.ndarray
        1D array of length n_radii containing the averaged profile across all lines and their mirrored halves.
    """
    Z_line = np.asarray(Z_line, dtype=float) 
    if Z_line.size == 0:
        return np.array([])
    Z_full = np.vstack([Z_line, Z_line[:, ::-1]])  # stacks original (+r) and mirrored (-r) arrays vertically, for which average is returned
    with np.errstate(all='ignore'):
        return np.nanmean(Z_full, axis=0) 

# SBW File Parsing and Cleaning
class sbwinfo(object):
    """
    Container for parsed SBW data
    """
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
# Parses .sbw file into sbwinfo object
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
    """
    Convert `sbwinfo` object (from `parsesbw`) into a cleaned dictionary format.

    Input
    ---
    sbwfile: sbwinfo (from `parsesbw`)
        Parsed SBW file object containing lot and wafer data.
    
    Output
    ---
    dict
        Dictionary with structure:
        {'Lot': str,
         'WaferData': {
             slot_key: {        (slot_key = WaferData[k])
                'SlotNo': str,
                'Lot': str,
                'Radius': list of float,
                'Angle': list of float,
                'Profiles': list of 2D lists [[float, float], ...]
                },...
    """
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

@st.cache_data(show_spinner=False) # Caches results of this function
def parsecleansbw(uploaded_bytes: bytes) -> Dict[str, Any]:
    """
    Parse (using `parsesbw`) and clean (using `cleansbw`) .sbw file uploaded by user, and return cleaned dict format

    Input
    ---
    uploaded_bytes: bytes
        Raw file content (from st.file_uploader).

    Output
    ---
    dict
        Cleaned dictionary in the same format as `cleansbw`, with structure:
          {'Lot': str,
         'WaferData': {
             slot_key: {        (slot_key = WaferData[k])
                'SlotNo': str,
                'Lot': str,
                'Radius': list of float,
                'Angle': list of float,
                'Profiles': list of 2D lists [[float, float], ...]
                },...
    """
    import tempfile
    obj = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sbw") as tmp: # create a temporary file .sbw
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

# Wafer Grid & Slot Caching
def Thkmatrix(wafer): 
    """
    Build 2D thickness matrix with rows = Angle, columns = Radius

    Input
    ---
    wafer: dict
        Wafer dictionary containing:
        - "Radius": list of float
        - "Angle": list of float
        - "Profiles": list of 2D arrays, each with columns
            [:,0] = Thickness values
            [:,1] = Flatness values (ignored in this function)

    Output
    --- 
    r: np.ndarray
        1D array of radii, shape (n_radius,).
    theta: np.ndarray
        1D array of angles, shape (n_theta,).
    Thk: np.ndarray
        2D array of thickness values, shape (n_theta, n_radius).
        Rows correspond to angles, columns correspond to radii.
    """
    r = np.asarray(wafer.get('Radius', []), dtype=float)
    theta = np.asarray(wafer.get('Angle', []), dtype=float)
    profiles = wafer.get('Profiles', [])
    nt, nr = len(theta), len(r)
    Thk = np.full((nt, nr), np.nan, dtype=float) # create 2D array filled with NaN, shape (nt, nr)
    for i in range(nt): # loop over each angle i and get corresponding Thk data at every r
        line = np.asarray(profiles[i], dtype=float) if i < len(profiles) else np.array([], dtype=float)
        if line.ndim == 2 and line.shape[1] > 0:
            Thk[i, :min(nr, line.shape[0])] = line[:nr, 0]
        else:
            Thk[i, :min(nr, line.size)] = line.ravel()[:nr]
    return r, theta, Thk

def Flatmatrix(wafer):
    """
    Build 2D flatness matrix with rows = Angle, columns = Radius

    Input
    ---
    wafer: dict
        Wafer dictionary containing:
        - "Radius": list of float
        - "Angle": list of float
        - "Profiles": list of 2D arrays, each with columns
            [:,0] = Thickness values (ignored in this function)
            [:,1] = Flatness values 

    Output
    --- 
    r: np.ndarray
        1D array of radii, shape (n_radius,).
    theta: np.ndarray
        1D array of angles, shape (n_theta,).
    Flat: np.ndarray
        2D array of thickness values, shape (n_theta, n_radius).
        Rows correspond to angles, columns correspond to radii.
    """
    r = np.asarray(wafer.get('Radius', []), dtype=float)
    theta = np.asarray(wafer.get('Angle', []), dtype=float)
    profiles = wafer.get('Profiles', [])
    nt, nr = len(theta), len(r)
    Flat = np.full((nt, nr), np.nan, dtype=float)
    for i in range(nt): # loop over each angle i and get corresponding Flat data at every r
        line = np.asarray(profiles[i], dtype=float) if i < len(profiles) else np.array([], dtype=float)
        if line.ndim == 2 and line.shape[1] > 1:
            Flat[i, :min(nr, line.shape[0])] = line[:nr, 1]
        else:
            Flat[i, :min(nr, line.size)] = line.ravel()[:nr]
    return r, theta, Flat

@dataclass
class SlotCache:
# Cache object containing precomputed arrays for each slot
    r: np.ndarray
    theta: np.ndarray
    Thk: np.ndarray
    Flat: np.ndarray
    Rmax: float
    X_mir: np.ndarray
    Y_mir: np.ndarray
    Thk_mir: np.ndarray
    Flat_mir: np.ndarray

def finite_max(arr: np.ndarray, default: float = 0.0) -> float:
    """
    Return max of finite values in array, default if empty.
    Used to find Rmax, which is used to draw wafer outline (line 430-439).

    Input
    ---
    arr: np.ndarray
        Input array.
    default: float
        Value to return if no finite values exist (default = 0.0).

    Output
    ---
    float
        Maximum finite value in the array, or the default = 0.0 if none exist.
    """
    af = arr[np.isfinite(arr)]
    return float(np.max(af)) if af.size else default

def build_slot_cache(wafer_dict) -> SlotCache:
    """
    Take wafer_dict and builds SlotCache

    Input
    ---
    wafer_dict: dict
        Wafer data dictionary (from cleansbw) containing:
        - 'Radius': list of float
            Radial positions (mm).
        - 'Angle': list of float
            Angular positions (radians).
        - 'Profiles': list of 2D arrays
            Each profile is shape (n_radius, 2), column 0 = Thickness, column 1 = Flatness.

    Output
    ---
    SlotCache
        Dataclass containing precomputed grids:
        - r: np.ndarray
            Radii (1D).
        - theta: np.ndarray
            Angles (1D).
        - Thk: np.ndarray, shape (n_theta, n_radius)
            Thickness grid.
        - Flat: np.ndarray, shape (n_theta, n_radius)
            Flatness grid.
        - Rmax: float
            Maximum finite radius (used for wafer outline).
        - X_mir, Y_mir: np.ndarray
            Cartesian grids after mirroring across wafer diameter.
        - Thk_mir, Flat_mir: np.ndarray
            Thickness/Flatness grids extended with mirrored halves to cover full 360°.
    """
    r, theta, Thk = Thkmatrix(wafer_dict) # Thickness matrix
    _, _, Flat = Flatmatrix(wafer_dict) # Flatness matrix
    Rmax = finite_max(r, 0.0)
    if theta.size and r.size:
        theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi)) # extends theta by mirroring it across wafer (theta + 180) and % 2pi ensures angles stay in [0,2pi]
        Thk_full = np.vstack([Thk, Thk[:, ::-1]]) if Thk.size else np.empty((0, 0)) # stacks original (+r) and mirrored (-r) arrays vertically
        Flat_full = np.vstack([Flat, Flat[:, ::-1]]) if Flat.size else np.empty((0, 0)) # stacks original (+r) and mirrored (-r) arrays vertically
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
        Rmax=Rmax, X_mir=X_mir, Y_mir=Y_mir, Thk_mir=Thk_full, Flat_mir=Flat_full
    )

@st.cache_data(show_spinner=False) # Caches results of this function
def cache_for_data(data: Dict[str, Any]) -> Dict[str, SlotCache]:
    """
    Build cache for all slots from cleaned wafer data.

    Input
    ---
    data: dict
        Cleaned wafer data dictionary (output of `cleansbw`), with structure:
        {'Lot': str,
         'WaferData': {
             slot_key: {        (slot_key = WaferData[k])
                'SlotNo': str,
                'Lot': str,
                'Radius': list of float,
                'Angle': list of float,
                'Profiles': list of 2D lists [[float, float], ...]
                },...

    Output
    ---
    dict of {str: SlotCache}
        Mapping from slot key (str) to SlotCache object containing precomputed matrices (thickness, flatness, mirrored data).
    """
    wafers = data.get('WaferData', {}) or {}
    return {k: build_slot_cache(w) for k, w in wafers.items()}


# Plot Utilities
def graph_arrays(c: SlotCache, graph: str):
    """
    Select thickness or flatness arrays from SlotCache.

    Input
    ---
    c: SlotCache
        Cached wafer slot data containing thickness and flatness matrices.
    graph: str
        - "flat": return flatness data
        - else: return thickness data

    Output
    ---
    tuple
        (line_array, surface_array, label)
        - line_array: np.ndarray
            Original (non-mirrored) 2D array, shape (n_theta, n_radius).
        - surface_array: np.ndarray
            Mirrored 2D array, shape (2*n_theta, n_radius), used for plotting full wafer surfaces.
        - label: str
            Axis label string ("Flatness (µm)" or "Thickness (µm)").
    """
    return (c.Flat, c.Flat_mir, 'Flatness (µm)') if graph == 'flat' else (c.Thk, c.Thk_mir, 'Thickness (µm)')

def graph_label(graph: str, prefix: str = "") -> str:
    """
    Graph title and label

    Input
    ---
    graph : str
        Graph type, either "flat" or "thk".
    prefix : str, optional
        Prefix to prepend ("PRE", "POST", "Average").

    Output
    ---
    str
        Label string ("Flatness", "Thickness", or with prefix, e.g., "PRE Flatness").
    """
    base = "Flatness" if graph == "flat" else "Thickness"
    if prefix:
        return f"{prefix} {base}"
    return f"{base}"

def robust_clip(Z: np.ndarray, p_lo: float, p_hi: float):
    """
    Clip values based on p_lo (lowest percentile) and p_hi (highest percentile).

    Input
    ---
    Z: np.ndarray
        Input array of values (e.g. thickness, flatness, or removal data).
    p_lo, p_hi: float
        Lowest percentile and highest percentile values        

    Output
    ---
    tuple
        (Z_clipped, vmin, vmax)
        - Z_clipped : np.ndarray
            Copy of Z with values clipped to [vmin, vmax].
        - vmin : float
            Lower bound (computed from p_lo).
        - vmax : float
            Upper bound (computed from p_hi).
    """
    Zf = Z[np.isfinite(Z)]
    if Zf.size == 0:
        return Z, 0.0, 1.0 # if Zf is empty, vmin=0.0 and vmax=1.0 to avoid crash
    vmin = float(np.nanpercentile(Zf, p_lo))
    vmax = float(np.nanpercentile(Zf, p_hi))
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = vmin + 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-9 # forces vmin >= vmin to avoid crash
    return np.clip(Z, vmin, vmax), vmin, vmax

def masknotch(Z: np.ndarray, k: float=4): # Outlier threshold = 4 
    """
    Mask notch (outliers) in array using Median Absolute Deviation (MAD).
    Any value further than k*MAD from median is replaced with NaN.

    Input
    ---
    Z : np.ndarray
        Input array of values (e.g. wafer thickness, flatness, or removal data).
    k : float, optional
        Threshold multiplier (set to 4).
        Any value further than k*MAD from the median is considered an outlier.

    Output
    ---
    np.ndarray
        Copy of input array with outliers replaced with NaN.

    """
    Zm = np.asarray(Z, dtype=float).copy()
    m = np.isfinite(Zm)
    if not m.any():
        return Zm
    Zf = Zm[m] # extract only finite values into Zf
    med = float(np.nanmedian(Zf))
    mad = float(np.nanmedian(np.abs(Zf - med))) * 1.4826 # 1.4826 = scaling factor to make MAD comparable to standard deviation
    if mad == 0 or not np.isfinite(mad):
        return Zm
    out = np.abs(Zm - med) > (k * mad) # Distance > k x MAD is marked as outlier -> NaN
    Zm[out] = np.nan
    return Zm

# Image Overlay (for Line Scanning Direction)
def overlay_images(base_url: str, overlay_url: str, overlay_size: int = 80, rotation_deg: float = 0.0) -> Image.Image:
# Overlay arrow on wafer image and rotate by rotation_deg
    base = Image.open(BytesIO(requests.get(base_url).content)).convert("RGBA") # waferimg
    overlay = Image.open(BytesIO(requests.get(overlay_url).content)).convert("RGBA") # arrowimg
    overlay = overlay.resize((overlay_size, overlay_size*3), Image.LANCZOS)
    overlay = overlay.rotate(rotation_deg, expand=True) # rotate arrow by rotation_deg set by user through select_slider
    w, h = base.size
    pos = ((w - overlay.width) // 2, (h - overlay.height) // 2)
    combined = base.copy()
    combined.paste(overlay, pos, overlay)
    return combined


# Plotting Functions
def plot_3d(X, Y, Z, zlabel: str, p_lo: float, p_hi: float, mask: bool, height: int = 600):
    """
    Inputs:
        X,Y,Z: 2D arrays of coordinates and surface values
        zlabel: str
            String label for z-axis and colorbar
        p_lo, p_hi: float
            Lowest percentile and highest percentile values
        mask: bool
            Mask outliers or not
        height: int
            Plot's height (in pixels)
    Output:
        3D plot in Streamlit
    """
    Z = np.asarray(Z)
    if Z.size == 0:
        return
    Zg = masknotch(Z) if mask else Z # Mask notch if mask checkbox is checked
    Zc, vmin, vmax = robust_clip(Zg, p_lo, p_hi) # vmin = lowest percentile; vmax = highest percentile
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Zg,
            surfacecolor=Zc, colorscale="Jet",
            cmin=vmin, cmax=vmax,
            colorbar=dict(title=zlabel, len=0.8, thickness=15),
            contours={
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlight": True,
                    "project": {"z": True},
                    "start": vmin,
                    "end": vmax,
                    "size": (vmax - vmin) / 20 if vmax > vmin else 1.0
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
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, autosize=True)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

def plot_2d(X, Y, Z, zlabel: str, radius_max: float, p_lo: float, p_hi: float, mask: bool, height: int=600):
    """
    Inputs:
        X,Y,Z: 2D arrays of coordinates and surface values
        zlabel: str
            String for colorbar
        radius_max: float
            Wafer radius
        p_lo, p_hi: float
            Lowest percentile and highest percentile values
        mask: bool
            Mask notch or not
        height: int
            Plot's height (in pixels)
    Output:
        2D plot in Streamlit
    """
    Z = np.asarray(Z)
    if Z.size == 0:
        return
    Zg = masknotch(Z) if mask else Z # Mask notch if mask checkbox is checked
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
    # wafer outline
    # theta = np.linspace(0, 2*np.pi, 200)
    # rmax = radius_max if np.isfinite(radius_max) and radius_max > 0 else 0.0
    # if rmax > 0:
    #     cx, cy = rmax * np.cos(theta), rmax * np.sin(theta)
    #     fig.add_trace(go.Scatter3d(
    #         x=cx, y=cy, z=[0.0]*cx.size,
    #         mode="lines", line=dict(color="black", width=2),
    #         showlegend=False
    #     ))
    fig.update_scenes(
        zaxis=dict(visible=False),
        xaxis_title="Radius (mm)",
        yaxis_title="Radius (mm)",
        aspectmode="data",
        camera=dict(eye=dict(x=0, y=0, z=11), up=dict(x=-1, y=0, z=0))
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode="pan", height=height, autosize=True)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

def plot_line_profile(r: np.ndarray, line: np.ndarray, zlabel: str, title: str, height: int = 500,
                      overlay_pre: Optional[np.ndarray] = None, overlay_post: Optional[np.ndarray] = None, avg=False,
                      waferimg: Optional[str] = None, rotation_deg: float = 0.0, positive_only: bool = False):
    """
    Inputs:
        r: np.ndarray
            1D array of radii
        line: np.ndarray
            1D array of values
        zlabel: str
            y-axis label
        title: str
        height: int
        overlay_pre, overlay_post: np.ndarray or None
            None by default
        avg: 
            True if Average Profile checkbox checked
        waferimg: str or None
            Wafer image
        rotation_deg: float
            Rotation for arrow image
        positive_only: bool
            if True, only plot r>=0
    Output:
        Line chart of a single angle in Streamlit
    """
    x = np.asarray(r, dtype=float)
    y = np.asarray(line, dtype=float)
    x_full = np.asarray(r, dtype=float)
    y_full = np.asarray(line, dtype=float)
    if positive_only: # for when Average Profile is selected, to chart only +r
        m = (x_full >= 0) & np.isfinite(y_full)
        x = x_full[m]
        y = y_full[m]
    else:
        x, y = finite_xy(-x_full, y_full) # else both +r and -r are charted
    fig = go.Figure()
    # overlay PRE
    if overlay_pre is not None: # if "Overlay line charts" checkbox is checked
        y_pre = np.asarray(overlay_pre, dtype=float)
        if positive_only and y_pre.size == y_full.size:
            y_pre = y_pre[m]
        y_pre = y_pre[:x.size]
        x_pre = x[:y_pre.size]
        x_pre, y_pre = finite_xy(x_pre, y_pre)
        if y_pre.size:
            fig.add_trace(go.Scatter(
                x=x_pre, y=y_pre, mode="lines",
                name="PRE", line=dict(width=1.0, color="lightgray")
            ))
    # overlay POST
    if overlay_post is not None: # if "Overlay line charts" checkbox is checked
        y_post = np.asarray(overlay_post, dtype=float)
        if positive_only and y_post.size == y_full.size:
            y_post = y_post[m]
        y_post = y_post[:x.size]
        x_post = x[:y_post.size]
        x_post, y_post = finite_xy(x_post, y_post)
        if y_post.size:
            fig.add_trace(go.Scatter(
                x=x_post, y=y_post, mode="lines",
                name="POST", line=dict(width=1.0, color="lightgray")
            ))
    # main line
    if y.size:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="red"),
            name="Removal"
        ))
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis_title="Radius (mm)",
        yaxis_title=zlabel,
        hovermode="x unified",
        dragmode="pan",
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=False)
    )
    if waferimg:
        col_plot, col_img = st.columns([12, 1])
        with col_plot:
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        with col_img:
            combined = overlay_images(
                "https://raw.githubusercontent.com/kaijwou/SBW-removal-profile/main/waferimg.jpg",
                "https://raw.githubusercontent.com/kaijwou/SBW-removal-profile/main/arrowimg.png",
                overlay_size=225,
                rotation_deg=rotation_deg
            )
            st.image(combined, width=200)
            st.markdown(f"<div style='text-align:center; font-size:0.9em; color:gray;'>{rotation_deg:.1f}°</div>", unsafe_allow_html=True)
    else:
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

def plot_line_grid(r: np.ndarray, theta: np.ndarray, Z_line: np.ndarray, zlabel: str,
                   nrows=2, ncols=4, height: int = 600,
                   overlay_pre: Optional[np.ndarray] = None, overlay_post: Optional[np.ndarray] = None, avg=False):
    """
    Inputs:
        r: np.ndarray
            1D radii
        theta: np.ndarray
            1D angles
        Z_line: np.ndarray
            2D array (n_theta, n_radii)
        zlabel: str
            y-axis label
        nrows, ncols: 
            subplot grid size
        height: int
        overlay_pre, overlay_post: np.ndarray or None
            optional PRE/POST overlays
        avg: 
            True if Average Profile selected
    Output:
        Line charts in Streamlit
    """
    r = np.asarray(r, dtype=float)
    Z_line = np.asarray(Z_line, dtype=float)
    if Z_line.size == 0:
        return
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=True)
    n = Z_line.shape[0]
    count = min(n, nrows*ncols)
    theta = np.asarray(theta, dtype=float)
    angs = np.degrees(theta[:count]) if theta.size else np.full(count, np.nan)
    has_pre = overlay_pre is not None and np.size(overlay_pre) != 0
    has_post = overlay_post is not None and np.size(overlay_post) != 0
    for i in range(count):
        row, col = i // ncols + 1, i % ncols + 1
        y = Z_line[i, :]
        x_i, y_i = finite_xy(-r, y)
        ang = angs[i] if i < angs.size else np.nan
        if has_pre:
            y_pre = overlay_pre[i, :] if i < np.asarray(overlay_pre).shape[0] else np.array([])
            x_pre, y_pre = finite_xy(r, y_pre)
            if y_pre.size:
                fig.add_trace(go.Scatter(x=x_pre, y=y_pre, mode="lines",
                                         line=dict(width=1.0, color="lightgray"),
                                         name="PRE"), row=row, col=col)
        if has_post:
            y_post = overlay_post[i, :] if i < np.asarray(overlay_post).shape[0] else np.array([])
            x_post, y_post = finite_xy(r, y_post)
            if y_post.size:
                fig.add_trace(go.Scatter(x=x_post, y=y_post, mode="lines",
                                         line=dict(width=1.0, color="lightgray"),
                                         name="POST"), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_i, y=y_i, mode="lines",
                                 line=dict(width=1.2, color="red"),
                                 showlegend=False,
                                 hovertemplate="x: %{x}<br>y: %{y}<extra></extra>"),
                      row=row, col=col)
        label = f"Angle {ang+180:.1f}°"
        fig.add_annotation(text=label, showarrow=False,
                           x=0.5, xref="x domain", y=1.15, yref="y domain",
                           font=dict(size=12, color="gray"), row=row, col=col)
        if row == nrows:
            fig.update_xaxes(title_text="Radius (mm)", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=zlabel, row=row, col=col)
    fig.update_layout(showlegend=False, dragmode="pan", height=height, margin=dict(l=30, r=30, t=60, b=30))
    for r_i in range(1, nrows+1):
        for c_i in range(1, ncols+1):
            fig.update_xaxes(matches="x1", row=r_i, col=c_i, showticklabels=True)
            fig.update_yaxes(matches="y1", row=r_i, col=c_i, showticklabels=True)
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def slot_options(data: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Return a list of Slot #'s for user to select in dropdown menu

    Intput
    ----------
    data: dict or None
        Cleaned wafer data dictionary (output of `cleansbw`) with structure:
        {'Lot': str,
         'WaferData': {
             slot_key: {        (slot_key = WaferData[k])
                'SlotNo': str,
                'Lot': str,
                'Radius': list of float,
                'Angle': list of float,
                'Profiles': list of 2D lists [[float, float], ...]
                },...

    Output
    -------
    list of tuple (str, str)
        A list of (display_label, slot_key) pairs
        - display_label: str
            Label shown in the dropdown  menu.
        - slot_key : str
            Internal key used to look up data in WaferData.
    """
    if not data:
        return []
    disp = []
    wafers = data.get('WaferData', {}) or {}
    for k in sort_keys(wafers):
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
    mask = st.checkbox("Mask notch", value=False)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    pre_file = st.file_uploader("Upload PRE .sbw", type=["sbw"], key="pre")
with colB:
    post_file = st.file_uploader("Upload POST .sbw", type=["sbw"], key="post")
with colC:
    graph = st.selectbox( # dropdown menu (Thickness | Flatness)
        "",
        options=[("Thickness", "thk"), ("Flatness", "flat")],
        format_func=lambda x: x[0]
    )[1]
    profile_mode = st.segmented_control("",["PRE", "POST", "REMOVAL"],label_visibility="hidden", width="stretch") # (PRE | POST | REMOVAL)
    avg_profiles = st.checkbox("Average Profile", key="avg_profiles", disabled=False)

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
        PRE_DATA = parsecleansbw(pre_file.read())
        PRE_CACHE = cache_for_data(PRE_DATA)
        st.success(f"Loaded {PRE_DATA.get('Lot', '')}")
    except Exception as e:
        st.error(f"Failed to parse PRE: {e}")

if post_file is not None:
    try:
        POST_DATA = parsecleansbw(post_file.read())
        POST_CACHE = cache_for_data(POST_DATA)
        st.success(f"Loaded {POST_DATA.get('Lot', '')}")
    except Exception as e:
        st.error(f"Failed to parse POST: {e}")

st.markdown("---")

# profile_mode == PRE or POST:
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
                            on_change=reset_plot, args=(plot_key,)) # on_change=reset_plot -> Changing slots resets session state, requiring Plot button to be pressed again

        sel_keys = [values[labels.index(lbl)] for lbl in sel] if sel else []
        if st.button("Plot", key=f"plot_btn_{profile_mode}"):
            st.session_state[plot_key] = True # Plot button as trigger

        prev_key = f"prev_avg_{profile_mode}"
        if prev_key not in st.session_state:
            st.session_state[prev_key] = avg_profiles
        elif st.session_state[prev_key] != avg_profiles:
            st.session_state[prev_key] = avg_profiles
            st.session_state[plot_key] = False

        if st.session_state.get(plot_key, False):
            if not sel_keys:
                st.warning("Choose at least one slot.")
            else:
                if avg_profiles:
                    for slot in sel_keys:
                        if slot not in cache:
                            st.warning(f"No cache for slot {slot}")
                            continue
                        c = cache[slot]
                        Z_line, _, zlabel = graph_arrays(c, graph)
                        if Z_line.size == 0:
                            st.warning(f"No data in slot {slot}")
                            continue

                        avg_profile = average_profile(Z_line)
                        if avg_profile.size == 0:
                            st.warning(f"No data in slot {slot}")
                            continue

                        X, Y = c.X_mir, c.Y_mir
                        nr = min(avg_profile.size, c.r.size)
                        Z_surf = np.tile(avg_profile[:nr], (X.shape[0], 1))[:, :nr]
                        X = X[:, :nr]
                        Y = Y[:, :nr]

                        lot = data.get('WaferData', {}).get(slot, {}).get('Lot', data.get('Lot', ''))
                        slotno = data.get('WaferData', {}).get(slot, {}).get('SlotNo', slot)
                        st.subheader(f"{graph_label(graph, 'Average')}\n{lot}({slotno})")

                        plot_line_profile(c.r[:nr], avg_profile[:nr], zlabel, "", height=520, avg=True, positive_only=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            plot_2d(X, Y, Z_surf, zlabel, c.Rmax, p_lo, p_hi, mask)
                        with col2:
                            plot_3d(X, Y, Z_surf, zlabel, p_lo, p_hi, mask)

                        st.markdown("---")
                else:
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
                            plot_2d(X, Y, Z_surf, zlabel, c.Rmax, p_lo, p_hi, mask)
                        with col2:
                            plot_3d(X, Y, Z_surf, zlabel, p_lo, p_hi, mask)

                        plot_line_grid(r, theta, Z_line, zlabel, nrows=2, ncols=4, height=600)

                        if len(theta) > 0:
                            angle_options = [f"{np.degrees(a)+180:.1f}°" for a in theta]
                            ang_key = f"ang_{profile_mode}_{slot}"
                            if ang_key not in st.session_state:
                                st.session_state[ang_key] = angle_options[0]
                            ang_str = st.select_slider("Angle", options=angle_options, key=ang_key)
                            idx = angle_options.index(ang_str)
                            ang = theta[idx]
                            line = Z_line[idx, :]
                            rotation_deg = float(ang_str.replace("°", ""))
                            plot_line_profile(r, line, zlabel, f"Angle {ang+180:.1f}°", height=520,
                                waferimg="https://raw.githubusercontent.com/kaijwou/SBW-removal-profile/main/waferimg.jpg", rotation_deg=rotation_deg)

                        st.markdown("---")

# profile_mode == REMOVAL:
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
                key="rem_pre_slots", on_change=reset_plot, args=(plot_key,)
            )
            pre_keys = [pre_values[pre_labels.index(lbl)] for lbl in sel_pre] if sel_pre else []
        with c2:
            sel_post = st.multiselect(
                "POST slots", post_labels, default=None,
                key="rem_post_slots", on_change=reset_plot, args=(plot_key,)
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
            
            if avg_profiles and n_pairs > 0:
                for pre_slot, post_slot in zip(pre_keys[:n_pairs], post_keys[:n_pairs]):
                    if pre_slot not in PRE_CACHE or post_slot not in POST_CACHE:
                        st.warning("Selected slot missing in cache.")
                        continue
                    A_c, B_c = PRE_CACHE[pre_slot], POST_CACHE[post_slot]
                    A_line, _, _ = graph_arrays(A_c, graph)
                    B_line, _, _ = graph_arrays(B_c, graph)
                    if A_line.size == 0 or B_line.size == 0:
                        st.warning("No overlapping data for removal.")
                        continue

                    nr = min(A_line.shape[1], B_line.shape[1], A_c.r.size, B_c.r.size)
                    if nr == 0:
                        st.warning("No overlapping data for removal.")
                        continue

                    A_avg = average_profile(A_line)[:nr]
                    B_avg = average_profile(B_line)[:nr]
                    Z_avg = A_avg - B_avg # Average PRE - Average POST

                    XA, YA = A_c.X_mir[:, :nr], A_c.Y_mir[:, :nr]
                    XB, YB = B_c.X_mir[:, :nr], B_c.Y_mir[:, :nr]
                    ZA = np.tile(A_avg, (XA.shape[0], 1))
                    ZB = np.tile(B_avg, (XB.shape[0], 1))
                    XZ, YZ = XA, YA
                    Zrem = np.tile(Z_avg, (XZ.shape[0], 1))

                    pre_lot = PRE_DATA.get('WaferData', {}).get(pre_slot, {}).get('Lot', PRE_DATA.get('Lot', ''))
                    post_lot = POST_DATA.get('WaferData', {}).get(post_slot, {}).get('Lot', POST_DATA.get('Lot', ''))
                    pre_slotno = PRE_DATA.get('WaferData', {}).get(pre_slot, {}).get('SlotNo', pre_slot)
                    post_slotno = POST_DATA.get('WaferData', {}).get(post_slot, {}).get('SlotNo', post_slot)
                    st.subheader(f"{graph_label(graph, 'Average')} Removal Profile\n{pre_lot}({pre_slotno}), {post_lot}({post_slotno})")

                    overlay_pre = A_avg if overlay_prepost_lines else None
                    overlay_post = B_avg if overlay_prepost_lines else None
                    plot_line_profile(
                        A_c.r[:nr], Z_avg, 'Removal (µm)', "",
                        height=520, avg=True,
                        overlay_pre=overlay_pre, overlay_post=overlay_post, positive_only=True
                    )

                    view_key = f"show3d_avg_pair_{pre_slot}_{post_slot}"
                    btn_key  = f"btn_avg_pair_{pre_slot}_{post_slot}"
                    if view_key not in st.session_state:
                        st.session_state[view_key] = False  # start with 2D
                    label = "◀" if st.session_state[view_key] else "▶"
                    if st.button(label, key=btn_key):
                        st.session_state[view_key] = not st.session_state[view_key]
                        st.rerun()

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.session_state[view_key]:
                            plot_3d(XA, YA, ZA, graph_label(graph, "PRE"), p_lo, p_hi, mask, height=300)
                        else:
                            plot_2d(XA, YA, ZA, graph_label(graph, "PRE"), A_c.Rmax, p_lo, p_hi, mask, height=300)

                    with c2:
                        if st.session_state[view_key]:
                            plot_3d(XB, YB, ZB, graph_label(graph, "POST"), p_lo, p_hi, mask, height=300)
                        else:
                            plot_2d(XB, YB, ZB, graph_label(graph, "POST"), B_c.Rmax, p_lo, p_hi, mask, height=300)

                    st.markdown("---")

            if not avg_profiles:
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
                    Z_line = A_line[:nt, :nr] - B_line[:nt, :nr] # PRE - POST
                    Z_surf = np.vstack([Z_line, Z_line[:, ::-1]])
                    theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi))
                    T, Rm = np.meshgrid(theta_full, r, indexing='ij')
                    X = Rm*np.cos(T)
                    Y = Rm*np.sin(T)
                    zlabel = 'Removal (µm)'

                    pre_lot = PRE_DATA.get('WaferData', {}).get(pre_slot, {}).get('Lot', PRE_DATA.get('Lot', ''))
                    post_lot = POST_DATA.get('WaferData', {}).get(post_slot, {}).get('Lot', POST_DATA.get('Lot', ''))
                    pre_slotno = PRE_DATA.get('WaferData', {}).get(pre_slot, {}).get('SlotNo', pre_slot)
                    post_slotno = POST_DATA.get('WaferData', {}).get(post_slot, {}).get('SlotNo', post_slot)

                    st.subheader(f"{graph_label(graph)} Removal Profile\n{pre_lot}({pre_slotno}), {post_lot}({post_slotno})")

                    c1, c2 = st.columns(2)
                    with c1:
                        rmax = float(np.max(r[np.isfinite(r)])) if np.isfinite(r).any() else 0.0
                        plot_2d(X, Y, Z_surf, zlabel, rmax, p_lo, p_hi, mask)
                    with c2:
                        view_key = f"show3d_{pre_slot}_{post_slot}"
                        if view_key not in st.session_state:
                            st.session_state[view_key] = False
                        label = "◀" if st.session_state[view_key] else "▶"
                        if st.button(label, key=f"btn_{pre_slot}_{post_slot}"):
                            st.session_state[view_key] = not st.session_state[view_key]
                            st.rerun()
                        if st.session_state[view_key]:
                            plot_3d(A_c.X_mir, A_c.Y_mir, A_surf, graph_label(graph, "PRE"), p_lo, p_hi, mask, height=300)
                            plot_3d(B_c.X_mir, B_c.Y_mir, B_surf, graph_label(graph, "POST"), p_lo, p_hi, mask, height=300)
                        else:
                            plot_2d(A_c.X_mir, A_c.Y_mir, A_surf, graph_label(graph, "PRE"), A_c.Rmax, p_lo, p_hi, mask, height=300)
                            plot_2d(B_c.X_mir, B_c.Y_mir, B_surf, graph_label(graph, "POST"), B_c.Rmax, p_lo, p_hi, mask, height=300)

                    overlay_pre = A_line[:nt, :nr] if overlay_prepost_lines else None
                    overlay_post = B_line[:nt, :nr] if overlay_prepost_lines else None

                    plot_line_grid(r, theta, Z_line, zlabel, nrows=2, ncols=4, height=600,
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
                        rotation_deg = float(ang_str.replace("°", ""))
                        pre_overlay_line = overlay_pre[idx, :] if overlay_pre is not None else None
                        post_overlay_line = overlay_post[idx, :] if overlay_post is not None else None
                        plot_line_profile(
                            r, line, zlabel, f"Angle {ang+180:.1f}°",
                            height=520,
                            overlay_pre=pre_overlay_line,
                            overlay_post=post_overlay_line,
                            waferimg="https://raw.githubusercontent.com/kaijwou/SBW-removal-profile/main/waferimg.jpg", rotation_deg=rotation_deg
                        )

                    st.markdown("---")
