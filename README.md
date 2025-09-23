# SBW Removal Profile — User Manual

**App type**: Streamlit

## Common Tasks

### View PRE or POST wafer profiles

1. Upload a PRE or POST .sbw file.
2. Select **`Thickness`** or **`Flatness`** from the dropdown menu.
3. Select **`PRE`** or **`POST`** in the segmented control.
4. Select one or more **Slots** from the multiselect dropdown menu. If multiple slots are selected, the plots are displayed in order.
5. (Optional) Check **`Mask notch`** in the sidebar to mask notch and filter out outlier values.
6. Click **Plot**.

### View REMOVAL wafer profile

1. Upload **both** PRE and POST .sbw files.
2. Select **`Thickness`** or **`Flatness`** from the dropdown menu.
3. Select **`REMOVAL`** in the segmented control.
4. Select PRE slots and POST slots. If counts differ, the slots are paired in order.
5. (Optional) Check **`Overlay line charts`** in the sidebar to show PRE/POST on top of REMOVAL line charts.
6. Click **Plot**.

### View average wafer profiles

1. Check **`Average Profile`**.
2. In PRE/POST profile mode: plots the average radial profile for the selected profile mode.
3. In REMOVAL profile mode: plots the average removal profile.

## User Interface

### Top controls

- Upload PRE .sbw | Upload POST .sbw: load .sbw files.
- **`Thickness | Flatness`** dropdown menu: select graph mode.
- **`PRE | POST | REMOVAL`** segmented control: select profile mode.
- **`Average Profile`** checkbox: switch to average profile mode.

### Sidebar — Display controls

- `Color clip low (%)`: slide to set the lowest percentile used for color range to prevent notch (outlier values) from skewing colors (default 0.5).
- `Color clip high (%)`: slide to set the highest percentile used for color range (default 100).
- `Mask notch`: replaces notch (outlier values) (beyond $k\times MAD$, default $k$ = 4) with NaN to prevent notch from skewing colors.
- (**`REMOVAL`** only) `Overlay line charts`: overlays PRE/POST lines on removal line plots.

### Angle selection

- **Angle** **slider**: slide to select an angle for a single-angle line chart.

The angle and direction at which the wafer has been line-scanned is indicated by the arrow shown on the icon of a wafer on top right of the chart.

### Controls and interactions

- **Hover**: shows x/y/z values.
- **Pan/Zoom**: use mouse to drag/scroll in plots.
- **Turnable rotation**: use mouse to drag and turn the surfaces.
- (**`REMOVAL`** only) **▶️** / **◀️ button**: to switch between 2D and 3D plots for PRE and POST.

# Code Explanation

### **`reset_plot`**

**A reset switch to control session state.** 

```python
def reset_plot(flag_key: str):
	st.session_state[flag_key] = False 
```

Streamlit reruns when the user interacts with the application (e.g., selecting new slots). `st.session_state[flag_key] = False` ensures that plotting is activated by the **Plot** button when the application reruns.

The function is used in `st.multiselect`, for example: 

```python
        ...
        plot_key = f"do_plot_{profile_mode}" 
        sel = st.multiselect("Slots", labels, default=None, key=f"{profile_mode}_slots",
                            on_change=reset_plot, args=(plot_key,))
```

Changing slot options resets session state, requiring the **Plot** button to be clicked again. `on_change=reset_plot` invokes `reset_plot()` function to be run whenever the widget's value (slots in this case) is changed. `arg=(plot_key,)` then passes `plot_key=f"do_plot_{profile_mode}"` as the argument to `reset_plot()`. As a result, when the widget’s value is changed, `st.session_state[plot_key]=False`. 

```python
        if st.button("Plot", key=f"plot_btn_{profile_mode}"):
            st.session_state[plot_key] = True
        ...
        
        if st.session_state.get(plot_key, False):
            if not sel_keys:
                st.warning("Choose at least one slot.")
            else:
                if avg_profiles:
```

Then, the lines of code above check if `st.session_state[plot_key]` is `True`. Plotting is activated only if slots have been selected and the **Plot** button has been clicked by the user. 

### `average_profile`

**Compute average radial profile by combining both +r and -r sides.**

```python
def average_profile(Z_line: np.ndarray) -> np.ndarray:
	Z_line = np.asarray(Z_line, dtype=float) 
  if Z_line.size == 0:
      return np.array([])
  Z_full = np.vstack([Z_line, Z_line[:, ::-1]]) 
  with np.errstate(all='ignore'):
      return np.nanmean(Z_full, axis=0) 
```

`Z_full = np.vstack([Z_line, Z_line[:, ::-1]])` stacks the original (+r) array and the mirrored (-r) array vertically. Then, the function returns the average of the stack. (`np.errstate(all='ignore')` suppresses error messages.)

### `Thkmatrix` & `Flatmatrix`

 **Build 2D thickness/flatness matrix with rows = Angle, columns = Radius**

```python
def Thkmatrix(wafer):
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
```

### `build_SlotCache`

**Take wafer_dict and build `SlotCache`.**

```python
def build_SlotCache(wafer_dict) -> SlotCache:
    r, theta, Thk = Thkmatrix(wafer_dict)
    _, _, Flat = Flatmatrix(wafer_dict)
    Rmax = finite_max(r, 0.0)
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
        Rmax=Rmax, X_mir=X_mir, Y_mir=Y_mir, Thk_mir=Thk_full, Flat_mir=Flat_full
    )
```

`theta_full = (np.concatenate([theta, theta + np.pi]) % (2*np.pi))` extends `theta` by mirroring it across the wafer `theta + np.pi` while `% (2*np.pi)` ensures that angles stay in the range $[0, 2\pi]$. Then, `Thk_full = np.vstack([Thk, Thk[:, ::-1]]) if Thk.size` stacks the original (+r) array and the mirrored (-r) array vertically (`::-1` reverses the sequence). 

This code uses the polar-coordinate identity:

$$
(r, \theta)\equiv(|r|, \theta+\pi)\{space}when\{space}r<0 
$$
