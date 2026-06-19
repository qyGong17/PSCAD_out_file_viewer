# Developer Notes & Architecture Guide

This document serves as context for LLMs (like Claude) and developers working on the `pscad_viewer.py` codebase.

## System Architecture

The application is a monolithic MVC-hybrid built with:
- **View/Controller**: `tkinter` (specifically `ttk` for modern Windows native widgets) and `matplotlib.backends.backend_tkagg`.
- **Model**: `pandas` DataFrames storing PSCAD time-series data, and dictionaries tracking Matplotlib lines/axes.

## Important Layout Quirks & Packing Orders

Tkinter's `pack` geometry manager is highly order-dependent. If you modify the layout, you must strictly adhere to the following packing sequences to avoid breaking the UI:

1. **Subplots Legend Panel vs. Main Plot Area**
   - **Problem**: The main plotting frame uses a `PanedWindow` that expands. If it is packed *before* the bottom legend panel, it will greedily consume all vertical space and push the legend panel off-screen.
   - **Rule**: `self._build_legend_panel()` (which packs `side=tk.BOTTOM`) MUST be called *before* `self._build_main_frame()` (which packs `side=tk.TOP, expand=True`).

2. **Matplotlib Canvas vs. Navigation Toolbar**
   - **Problem**: The `FigureCanvasTkAgg` wants to expand. If it is packed before the `NavigationToolbar2Tk` is instantiated, it pushes the toolbar out of bounds.
   - **Rule**: You MUST create the `NavigationToolbar2Tk` object *before* calling `canvas.get_tk_widget().pack(..., expand=True)`.

3. **PanedWindow Resizing**
   - **Problem**: The right-hand signals sidebar used to expand uncontrollably when the window was maximized.
   - **Rule**: In `paned.add(...)`, explicitly assign `stretch="always"` to the plot frame, and `stretch="never"` to the signals frame.

## Performance Optimizations

1. **Background Loading (`_read_pscad_folder`)**
   - PSCAD outputs can be gigabytes in size. To prevent UI freezing, `_load_folder_thread` runs in a `daemon=True` background thread.
   - *Strict Rule*: Do not update Tkinter widgets directly from this thread. Use `self.after(0, callback, data)` to safely pass the parsed DataFrame back to the main thread.
   - The parallel file reading uses a fast `pd.read_csv(engine="c")` combined with `ThreadPoolExecutor`.
   - After parsing, the dataframe is automatically cached to `[project].csv`. 

2. **Y-Axis Zoom-to-Fit (`_on_key_press`)**
   - Pressing `Y` computes the min/max of traces strictly within the currently visible X-window limits.
   - Because time arrays can contain millions of points, using boolean masks `ydata[(xdata > xmin) & (xdata < xmax)]` is dangerously slow.
   - **Optimization**: Because PSCAD outputs guarantee that the `time` column is strictly monotonically increasing, we use `numpy.searchsorted(xdata, limit)` to execute an $O(\log N)$ binary search, making the masking slice practically instantaneous.

## Visual Design Choices

- **Modern ttk**: Avoid raw `tk.Button`, `tk.Frame`, etc. Always use `ttk.Button`, `ttk.LabelFrame` so that the app automatically adopts the `vista` or `clam` Windows themes.
- **Y-Label Alignment**: When Matplotlib Y-tick widths jump from single digits to scientific notation, the Y-label is pushed horizontally. We counter this by invoking `self._fig.align_ylabels(self._axes)` after adding/removing traces, which forces all Y-labels to strictly align vertically across all subplots.
