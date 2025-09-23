# User Manual

**App type**: Streamlit

## Common Tasks

### View PRE or POST wafer profiles

1. Upload a PRE or POST .sbw file.
2. Select **`Thickness`** or **`Flatness`** from the dropdown menu.
3. Select **`PRE`** or **`POST`** in the segmented control.
4. Select one or more **Slots** from the multiselect dropdown menu. If multiple slots are selected, the plots are displayed in order.
5. (Optional) Check **`Mask notch`** in the sidebar to mask notch and filter out outlier values.
6. Click **`Plot`**.

### View REMOVAL wafer profile

1. Upload **both** PRE and POST .sbw files.
2. Select **`Thickness`** or **`Flatness`** from the dropdown menu.
3. Select **`REMOVAL`** in the segmented control.
4. Select PRE slots and POST slots. If counts differ, the slots are paired in order.
5. (Optional) Check **`Overlay line charts`** in the sidebar to show PRE/POST on top of REMOVAL line charts.
6. Click **`Plot`**.

### View average profiles

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
- **Pan/Zoom**: use the mouse to drag/scroll in plots.
- **Turnable rotation**: use the mouse to drag and turn the surfaces.
- (**`REMOVAL`** only) **▶️ / ◀️ button**: to switch between 2D and 3D plots for PRE and POST.
