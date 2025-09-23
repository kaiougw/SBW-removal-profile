# User Manual

**App type**: Streamlit

## Common Tasks

### A. View PRE or POST wafer profiles

1. Upload a PRE or POST .sbw file.
2. Select **`Thickness`** or **`Flatness`** from the dropdown menu.
3. Select **`PRE`** or **`POST`** in the segmented control.
4. Select one or more **Slots** from the multiselect dropdown menu. If multiple slots are selected, the plots are displayed in order.
5. (Optional) Check **`Mask notch`** in the sidebar to mask notch and filter out outlier values.
6. Click **Plot**.

### B. View REMOVAL wafer profile

1. Upload **both** PRE and POST .sbw files.
2. Select **`Thickness`** or **`Flatness`** from the dropdown menu.
3. Select **`REMOVAL`** in the segmented control.
4. Select PRE slots and POST slots. If counts differ, the slots are paired in order.
5. (Optional) Check **`Overlay line charts`** in the sidebar to show PRE/POST on top of REMOVAL line charts.
6. Click **Plot**.

### C. View average wafer profiles

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

## Code Explanation

### **`reset_plot`**

**A reset switch to control session state.** 

```jsx
def reset_plot(flag_key: str):
	st.session_state[flag_key] = False 
```

Streamlit reruns when the user interacts with the application (e.g., selecting new slots to be plotted). `st.session_state[flag_key] = False` ensures that plotting is activated by the **Plot** button when the application reruns.

The function is used in `st.multiselect` , for example: 

```python
        ...
        plot_key = f"do_plot_{profile_mode}" 
        sel = st.multiselect("Slots", labels, default=None, key=f"{profile_mode}_slots",
                            on_change=reset_plot, args=(plot_key,))
```

Changing slot options resets session state, requiring the **Plot** button to be pressed again. `on_change=reset_plot` invokes `reset_plot()` function to be run whenever the widget's value (slots in this case) is changed. `arg=(plot_key,)` then passes `plot_key=f"do_plot_{profile_mode}"` as the argument to `reset_plot()`. As a result, when 

### `average_profile`

**Compute average radial profile by combining both +r and -r sides.**

```jsx
def average_profile(Z_line: np.ndarray) -> np.ndarray:
	Z_line = np.asarray(Z_line, dtype=float) 
  if Z_line.size == 0:
      return np.array([])
  Z_full = np.vstack([Z_line, Z_line[:, ::-1]]) 
  with np.errstate(all='ignore'):
      return np.nanmean(Z_full, axis=0) 
```

**Input**

**Z_line: np.ndarray**

**2D array, shape (n_lines, n_radii), where each row is a scan line along radius positions.**

**Output**

**np.ndarray**

**1D array of length n_radii containing the averaged profile across all lines and their mirrored halves.**

`Z_full = np.vstack([Z_line, Z_line[:, ::-1]])`