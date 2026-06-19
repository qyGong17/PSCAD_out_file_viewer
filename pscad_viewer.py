#! python3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import re
import json
from math import ceil
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cycler as _cycler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

plt.rcParams.update({
    "font.family":                   "serif",
    "font.serif":                    ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":              "stix",
    "axes.prop_cycle":               _cycler(color=["0072BD", "D95319", "EDB120", "7E2F8E", "77AC30"]),
    "lines.linewidth":               1.0,
    "lines.markersize":              3,
    "axes.linewidth":                0.8,
    "axes.spines.top":               True,
    "axes.spines.right":             True,
    "xtick.direction":               "in",
    "ytick.direction":               "in",
    "xtick.major.size":              3,
    "ytick.major.size":              3,
    "legend.frameon":                False,
    "legend.loc":                    "best",
    "grid.linestyle":                ":",
    "grid.linewidth":                0.5,
    "grid.alpha":                    0.7,
    "figure.constrained_layout.use": True,
})

def _read_pscad_folder(path: str, project: str) -> pd.DataFrame:
    """Read a PSCAD .if18 output folder directly into a DataFrame.

    Bypasses ImPSCAD.PSCADVar to avoid the write-then-reread CSV round-trip
    and the slow pairwise reduce(merge) it uses.  Falls back to a cached CSV
    if one exists and is newer than all .out files.
    """
    csv_path = os.path.join(path, project + ".csv")
    out_paths = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if re.match(rf"^{re.escape(project)}_\d+\.out$", f)
    )

    # Cache hit: existing CSV is newer than all .out files
    if out_paths and os.path.exists(csv_path):
        csv_mtime = os.path.getmtime(csv_path)
        if csv_mtime > max(os.path.getmtime(p) for p in out_paths):
            return pd.read_csv(csv_path)

    # Parse variable names from .inf
    inf_path = os.path.join(path, project + ".inf")
    with open(inf_path) as f:
        text = f.read()
    headers = re.findall(r'Desc="(.*?)"', text)
    headers = [h.replace(" ", "_") for h in headers]

    if not out_paths:
        raise FileNotFoundError(f"No .out files found in {path}")

    # Read all .out files in parallel (I/O bound)
    def _read(p):
        return pd.read_csv(p, sep=r"\s+", header=None, engine="c")

    with ThreadPoolExecutor() as pool:
        frames = list(pool.map(_read, out_paths))

    # All files share the same time axis — concat columns directly (no merge)
    time_col = frames[0].iloc[:, 0]
    expected_len = len(time_col)
    for i, f in enumerate(frames):
        if len(f) != expected_len:
            raise ValueError(f"File {out_paths[i]} has a different number of rows ({len(f)}) than expected ({expected_len}).")
            
    data_cols = pd.concat([f.iloc[:, 1:] for f in frames], axis=1)
    df = pd.concat([time_col, data_cols], axis=1)
    df.columns = ["time"] + headers
    
    # Auto-cache to CSV for future loading
    df.to_csv(csv_path, index=False)
    
    return df


FS_LABEL  = 15
FS_TICK   = 15
FS_LEGEND = 15


class PSCADViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PSCAD Waveform Viewer")
        self.geometry("1200x800")

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        self._runs = {}                # dict of {run_id: DataFrame}
        self._active_run = tk.StringVar()
        self._run_counter = 0
        self._subplot_data = []        # list of dicts, one per subplot
        self._color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._color_idx = 0
        self._active_sp = tk.StringVar(value="Subplot 1")

        self._build_toolbar()
        self._build_legend_panel()
        self._build_main_frame()
        self._init_subplots(1)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        plt.close(self._fig)
        self.quit()
        self.destroy()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=4)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open Folder…", command=self._browse_folder, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Save CSV…",    command=self._save_csv,      width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Clear All",    command=self._clear_all,     width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Save Session", command=self._save_session,  width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Load Session", command=self._load_session,  width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Math...",      command=self._show_math_dialog, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="FFT",          command=self._show_fft_dialog, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Frame(bar, width=16).pack(side=tk.LEFT)
        ttk.Label(bar, text="Cursors: Ctrl+LClick/RClick").pack(side=tk.LEFT, padx=6)
        ttk.Frame(bar, width=16).pack(side=tk.LEFT)
        ttk.Button(bar, text="Add Subplot",  command=self._add_subplot,   width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(bar, text="Active:").pack(side=tk.LEFT, padx=(12, 2))
        self._sp_menu = ttk.Combobox(bar, textvariable=self._active_sp, state="readonly", width=12)
        self._sp_menu.set("Subplot 1")
        self._sp_menu.pack(side=tk.LEFT, padx=2)

    def _build_main_frame(self):
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, minsize=600, stretch="always")

        self._fig = plt.figure(figsize=(9, 6))
        self._axes = []

        canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        canvas.draw()
        
        nav = NavigationToolbar2Tk(canvas, plot_frame)
        nav.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._canvas = canvas
        self._canvas.mpl_connect('key_press_event', self._on_key_press)
        self._canvas.mpl_connect('button_press_event', self._on_click)

        self._cursor_a_x = None
        self._cursor_b_x = None
        self._cursor_lines_a = []
        self._cursor_lines_b = []
        self._cursor_label = None

        sig_frame = ttk.LabelFrame(paned, text="Signals", padding=4)
        paned.add(sig_frame, minsize=180, stretch="never")

        run_frame = ttk.Frame(sig_frame)
        run_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Label(run_frame, text="Run:").pack(side=tk.LEFT)
        self._run_menu = ttk.Combobox(run_frame, textvariable=self._active_run, state="readonly")
        self._run_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self._run_menu.bind("<<ComboboxSelected>>", self._on_run_selected)

        self._search_var = tk.StringVar()
        search_entry = ttk.Entry(sig_frame, textvariable=self._search_var)
        search_entry.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self._search_var.trace_add("write", self._filter_signals)

        sb = ttk.Scrollbar(sig_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(sig_frame, yscrollcommand=sb.set,
                                   selectmode=tk.EXTENDED, width=25)
        sb.config(command=self._listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._listbox.bind("<Double-Button-1>", lambda e: self._add_to_subplot())

        ttk.Button(sig_frame, text="Add to subplot →",
                  command=self._add_to_subplot).pack(side=tk.BOTTOM, fill=tk.X, pady=4)

    def _build_legend_panel(self):
        outer = ttk.LabelFrame(self, text="Subplots", padding=4)
        outer.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

        canvas = tk.Canvas(outer, height=90)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        self._legend_outer = inner
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

    # ── Subplot management ────────────────────────────────────────────────────
    def _init_subplots(self, n):
        for sp in self._subplot_data:
            if sp.get("_row_outer"):
                sp["_row_outer"].destroy()
        self._subplot_data.clear()
        for i in range(n):
            sp_dict = {"ax": None, "traces": {}, "legend_frame": None,
                       "legend_rows": {}, "_row_outer": None}
            self._subplot_data.append(sp_dict)
            self._add_subplot_legend_frame(i)
        self._rebuild_axes()
        self._refresh_sp_menu()

    def _add_subplot(self):
        i = len(self._subplot_data)
        sp_dict = {"ax": None, "traces": {}, "legend_frame": None,
                   "legend_rows": {}, "_row_outer": None}
        self._subplot_data.append(sp_dict)
        self._add_subplot_legend_frame(i)
        self._rebuild_axes()
        self._refresh_sp_menu()
        self._active_sp.set(f"Subplot {i + 1}")

    def _remove_subplot(self, sp_idx):
        if len(self._subplot_data) <= 1:
            messagebox.showwarning("Remove Subplot", "Cannot remove the last subplot.")
            return
        self._subplot_data[sp_idx]["_row_outer"].destroy()
        del self._subplot_data[sp_idx]
        for i, sp in enumerate(self._subplot_data):
            sp["legend_frame"].config(text=f"Subplot {i + 1}")
        self._rebuild_axes()
        self._refresh_sp_menu()

    def _add_subplot_legend_frame(self, sp_idx):
        row_outer = ttk.Frame(self._legend_outer)
        row_outer.pack(side=tk.TOP, fill=tk.X, pady=1)

        frame = ttk.LabelFrame(row_outer, text=f"Subplot {sp_idx + 1}", padding=2)
        frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        sp = self._subplot_data[sp_idx]
        sp["legend_frame"] = frame
        sp["_row_outer"] = row_outer

        # Y-axis label entry
        ylabel_var = tk.StringVar(value=f"SP {sp_idx + 1}")
        sp["ylabel"] = ylabel_var
        ylabel_host = ttk.Frame(frame)
        ylabel_host.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(ylabel_host, text="Y label:").pack(side=tk.LEFT)
        entry = ttk.Entry(ylabel_host, textvariable=ylabel_var, width=10)
        entry.pack(side=tk.LEFT)

        def _apply_ylabel(e=None, f=frame, var=ylabel_var):
            for j, s in enumerate(self._subplot_data):
                if s["legend_frame"] is f:
                    s["ax"].set_ylabel(var.get(), fontsize=FS_LABEL)
                    if self._axes:
                        self._fig.align_ylabels(self._axes)
                    self._canvas.draw_idle()
                    return

        entry.bind("<Return>",   _apply_ylabel)
        entry.bind("<FocusOut>", _apply_ylabel)

        def _remove(f=frame):
            for j, s in enumerate(self._subplot_data):
                if s["legend_frame"] is f:
                    self._remove_subplot(j)
                    return

        ttk.Button(row_outer, text="Remove SP", command=_remove, width=10).pack(side=tk.LEFT, padx=4)

    def _rebuild_axes(self):
        n = len(self._subplot_data)
        self._fig.clear()
        self._axes = []
        gs = self._fig.add_gridspec(n, 1, hspace=0.15)
        for i in range(n):
            ax = self._fig.add_subplot(gs[i, 0], sharex=self._axes[0] if i > 0 else None)
            ylabel = self._subplot_data[i].get("ylabel")
            ax.set_ylabel(ylabel.get() if ylabel else f"SP {i + 1}", fontsize=FS_LABEL)
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(True)
            self._axes.append(ax)
            self._subplot_data[i]["ax"] = ax
        if n > 0:
            self._axes[-1].set_xlabel("Time (s)", fontsize=FS_LABEL)
            self._fig.align_ylabels(self._axes)
        for i, sp in enumerate(self._subplot_data):
            for trace_id, trace in sp["traces"].items():
                run_id = trace["run_id"]
                col = trace["col"]
                df = self._runs.get(run_id)
                if df is not None:
                    t_arr = df["time"].to_numpy()
                    line, = self._axes[i].plot(t_arr, df[col].to_numpy(),
                                               color=trace["color"], label=trace_id)
                    trace["line"] = line
        self._canvas.draw_idle()

    def _refresh_sp_menu(self):
        options = [f"Subplot {i + 1}" for i in range(len(self._subplot_data))]
        if hasattr(self, '_sp_menu') and isinstance(self._sp_menu, ttk.Combobox):
            self._sp_menu['values'] = options
            if self._active_sp.get() not in options and options:
                self._sp_menu.set(options[0])
                self._active_sp.set(options[0])

    # ── Data loading ──────────────────────────────────────────────────────────
    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select PSCAD output folder (.if18)")
        if path:
            self._load_folder_bg(path)

    def _load_folder_bg(self, path):
        self._loading_win = tk.Toplevel(self)
        self._loading_win.title("Loading...")
        self._loading_win.geometry("300x100")
        self._loading_win.transient(self)
        self._loading_win.grab_set()
        
        ttk.Label(self._loading_win, text="Reading PSCAD files...").pack(pady=10)
        progress = ttk.Progressbar(self._loading_win, mode='indeterminate')
        progress.pack(fill=tk.X, padx=20, pady=10)
        progress.start()
        
        threading.Thread(target=self._load_folder_thread, args=(path,), daemon=True).start()

    def _load_folder_thread(self, path):
        name = os.path.basename(path)
        project = name.replace(".if18", "")
        try:
            df = _read_pscad_folder(path, project)
            self.after(0, self._on_load_success, df, project)
        except Exception as exc:
            self.after(0, self._on_load_error, str(exc))

    def _on_load_success(self, df, project):
        if hasattr(self, '_loading_win') and self._loading_win.winfo_exists():
            self._loading_win.destroy()
            
        self._run_counter += 1
        run_id = f"Run {self._run_counter} ({project})"
        self._runs[run_id] = df
        
        options = list(self._runs.keys())
        self._run_menu['values'] = options
        self._active_run.set(run_id)
        self._populate_signals(df)

    def _on_load_error(self, err_msg):
        if hasattr(self, '_loading_win') and self._loading_win.winfo_exists():
            self._loading_win.destroy()
        messagebox.showerror("Load error", err_msg)
        
    def _load_folder(self, path):
        self._load_folder_bg(path)

    def _on_run_selected(self, event=None):
        run_id = self._active_run.get()
        if run_id in self._runs:
            self._populate_signals(self._runs[run_id])

    def _populate_signals(self, df):
        self._all_signals = sorted(c for c in df.columns if c.lower() != "time")
        self._filter_signals()

    def _filter_signals(self, *args):
        self._listbox.delete(0, tk.END)
        query = ""
        if hasattr(self, '_search_var'):
            query = self._search_var.get().lower()
            
        if not hasattr(self, '_all_signals'):
            return
            
        for col in self._all_signals:
            if query in col.lower():
                self._listbox.insert(tk.END, col)

    # ── Trace management ──────────────────────────────────────────────────────
    def _active_sp_index(self):
        label = self._active_sp.get()
        for i in range(len(self._subplot_data)):
            if label == f"Subplot {i + 1}":
                return i
        return 0

    def _next_color(self):
        c = self._color_cycle[self._color_idx % len(self._color_cycle)]
        self._color_idx += 1
        return c

    def _add_to_subplot(self):
        run_id = self._active_run.get()
        if not run_id or run_id not in self._runs:
            messagebox.showwarning("No data", "Load a PSCAD folder first.")
            return
        selection = self._listbox.curselection()
        if not selection:
            return
        sp_idx = self._active_sp_index()
        for i in selection:
            col = self._listbox.get(i)
            self._add_trace(sp_idx, run_id, col)

    def _add_trace(self, sp_idx, run_id, col):
        sp = self._subplot_data[sp_idx]
        trace_id = f"{run_id} : {col}"
        if trace_id in sp["traces"]:
            return
        df = self._runs[run_id]
        color = self._next_color()
        t_arr = df["time"].to_numpy()
        line, = sp["ax"].plot(t_arr, df[col].to_numpy(), color=color, label=trace_id)
        sp["ax"].relim()
        sp["ax"].autoscale_view()
        sp["traces"][trace_id] = {"line": line, "color": color, "run_id": run_id, "col": col}
        self._add_legend_row(sp_idx, trace_id, color)
        if self._axes:
            self._fig.align_ylabels(self._axes)
        self._canvas.draw_idle()
        self._canvas.toolbar.update()

    def _remove_trace(self, sp_idx, trace_id):
        sp = self._subplot_data[sp_idx]
        if trace_id not in sp["traces"]:
            return
        sp["traces"][trace_id]["line"].remove()
        del sp["traces"][trace_id]
        row = sp["legend_rows"].pop(trace_id, None)
        if row:
            row.destroy()
        sp["ax"].relim()
        sp["ax"].autoscale_view()
        if self._axes:
            self._fig.align_ylabels(self._axes)
        self._canvas.draw_idle()
        self._canvas.toolbar.update()

    def _add_legend_row(self, sp_idx, col, color):
        sp = self._subplot_data[sp_idx]
        frame = sp["legend_frame"]
        row = ttk.Frame(frame)
        row.pack(side=tk.LEFT, padx=4)

        swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0)
        swatch.create_rectangle(2, 2, 14, 14, fill=color, outline="")
        swatch.pack(side=tk.LEFT)

        ttk.Label(row, text=col).pack(side=tk.LEFT, padx=2)

        def _do_remove(f=frame, c=col):
            for j, s in enumerate(self._subplot_data):
                if s["legend_frame"] is f:
                    self._remove_trace(j, c)
                    return

        ttk.Button(row, text="✕", width=2, command=_do_remove).pack(side=tk.LEFT)

        sp["legend_rows"][col] = row

    # ── Save / Clear ──────────────────────────────────────────────────────────
    def _save_csv(self):
        run_cols = {}
        for sp in self._subplot_data:
            for trace_id, trace in sp["traces"].items():
                r_id = trace["run_id"]
                c = trace["col"]
                if r_id not in run_cols:
                    run_cols[r_id] = set()
                run_cols[r_id].add(c)
                
        if not run_cols:
            messagebox.showwarning("Nothing to save", "Add signals to subplots first.")
            return
        
        path = filedialog.asksaveasfilename(
            title="Save as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
            
        try:
            master_df = pd.DataFrame()
            for r_id, cols in run_cols.items():
                df = self._runs[r_id][["time"] + list(cols)].copy()
                if len(run_cols) > 1:
                    rename_map = {c: f"[{r_id}] {c}" for c in cols}
                    df = df.rename(columns=rename_map)
                
                if master_df.empty:
                    master_df = df
                else:
                    master_df = pd.merge(master_df, df, on="time", how="outer")
            
            master_df = master_df.sort_values("time")
            master_df.to_csv(path, index=False)
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _save_session(self):
        if not self._runs:
            messagebox.showwarning("No data", "Load a PSCAD folder first.")
            return
            
        path = filedialog.asksaveasfilename(
            title="Save Session",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
            
        session_data = []
        for sp in self._subplot_data:
            sp_info = {"ylabel": sp["ylabel"].get(), "traces": []}
            for trace_id, trace in sp["traces"].items():
                sp_info["traces"].append({"col": trace["col"], "color": trace["color"], "run_id": trace["run_id"]})
            session_data.append(sp_info)
            
        try:
            with open(path, "w") as f:
                json.dump(session_data, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _load_session(self):
        if not self._runs:
            messagebox.showwarning("No data", "Load a PSCAD folder first.")
            return
            
        path = filedialog.askopenfilename(
            title="Load Session",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
            
        try:
            with open(path, "r") as f:
                session_data = json.load(f)
                
            self._clear_all()
            self._init_subplots(len(session_data))
            
            for i, sp_info in enumerate(session_data):
                self._subplot_data[i]["ylabel"].set(sp_info.get("ylabel", f"SP {i+1}"))
                for tr in sp_info.get("traces", []):
                    col = tr.get("col")
                    color = tr.get("color")
                    run_id = tr.get("run_id")
                    
                    if not run_id or run_id not in self._runs or col not in self._runs[run_id].columns:
                        continue
                        
                    try:
                        self._color_idx = self._color_cycle.index(color)
                    except ValueError:
                        pass
                        
                    self._add_trace(i, col)
                    
            self._rebuild_axes()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _clear_all(self):
        for sp_idx, sp in enumerate(self._subplot_data):
            for col in list(sp["traces"]):
                self._remove_trace(sp_idx, col)

    def _show_math_dialog(self):
        run_id = self._active_run.get()
        if not run_id or run_id not in self._runs:
            messagebox.showwarning("No data", "Load data first.")
            return
            
        win = tk.Toplevel(self)
        win.title("Trace Math (Active Run)")
        win.geometry("400x150")
        win.transient(self)
        
        ttk.Label(win, text="New Signal Name:").pack(pady=(10, 0))
        name_var = tk.StringVar()
        ttk.Entry(win, textvariable=name_var).pack(fill=tk.X, padx=10)
        
        ttk.Label(win, text="Expression (e.g. df['Va'] * 2):").pack(pady=(10, 0))
        expr_var = tk.StringVar()
        ttk.Entry(win, textvariable=expr_var).pack(fill=tk.X, padx=10)
        
        def _apply():
            name = name_var.get()
            expr = expr_var.get()
            if not name or not expr:
                return
            try:
                df = self._runs[run_id]
                local_env = {"df": df, "np": np}
                res = eval(expr, {}, local_env)
                df[name] = res
                self._populate_signals(df)
                win.destroy()
            except Exception as e:
                messagebox.showerror("Math Error", str(e))
                
        ttk.Button(win, text="Calculate", command=_apply).pack(pady=10)

    def _show_fft_dialog(self):
        run_id = self._active_run.get()
        if not run_id or run_id not in self._runs:
            return
            
        selection = self._listbox.curselection()
        if not selection:
            messagebox.showwarning("FFT", "Select a signal from the list first.")
            return
            
        col = self._listbox.get(selection[0])
        sp_idx = self._active_sp_index()
        ax = self._subplot_data[sp_idx]["ax"]
        if ax is None:
            return
            
        df = self._runs[run_id]
        xmin, xmax = ax.get_xlim()
        xdata = df["time"].to_numpy()
        ydata = df[col].to_numpy()
        
        idx_min = np.searchsorted(xdata, xmin, side='left')
        idx_max = np.searchsorted(xdata, xmax, side='right')
        
        if idx_min >= idx_max:
            return
            
        t = xdata[idx_min:idx_max]
        y = ydata[idx_min:idx_max]
        
        N = len(y)
        if N < 2:
            return
            
        T = (t[-1] - t[0]) / N if N > 1 else 1.0
        yf = np.fft.fft(y)
        xf = np.fft.fftfreq(N, T)[:N//2]
        
        win = tk.Toplevel(self)
        win.title(f"FFT Analysis: {col}")
        win.geometry("600x400")
        
        fig = plt.figure(figsize=(6, 4))
        ax_fft = fig.add_subplot(111)
        ax_fft.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax_fft.set_xlim(0, max(600, xf[-1] if len(xf) > 0 else 600))
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")
        ax_fft.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        
        nav = NavigationToolbar2Tk(canvas, win)
        nav.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_click(self, event):
        if not event.inaxes: return
        if event.key == 'control':
            if event.button == 1: # Left click
                for l in self._cursor_lines_a: l.remove()
                self._cursor_lines_a.clear()
                self._cursor_a_x = event.xdata
                for ax in self._axes:
                    self._cursor_lines_a.append(ax.axvline(event.xdata, color='r', linestyle='--'))
                self._update_cursor_text()
                self._canvas.draw_idle()
            elif event.button == 3: # Right click
                for l in self._cursor_lines_b: l.remove()
                self._cursor_lines_b.clear()
                self._cursor_b_x = event.xdata
                for ax in self._axes:
                    self._cursor_lines_b.append(ax.axvline(event.xdata, color='b', linestyle='--'))
                self._update_cursor_text()
                self._canvas.draw_idle()

    def _update_cursor_text(self):
        if self._cursor_a_x is not None and self._cursor_b_x is not None:
            dx = abs(self._cursor_b_x - self._cursor_a_x)
            freq = 1.0/dx if dx > 0 else 0
            text = f"ΔX: {dx:.4f}s | Freq: {freq:.2f}Hz"
            if self._cursor_label is None:
                self._cursor_label = ttk.Label(self._legend_outer, text=text, font=("Consolas", 12, "bold"), foreground="blue")
                self._cursor_label.pack(side=tk.BOTTOM, pady=4)
            else:
                self._cursor_label.config(text=text)

    def _on_key_press(self, event):
        if event.key and event.key.lower() == 'y' and event.inaxes:
            ax = event.inaxes
            xmin, xmax = ax.get_xlim()
            
            y_min = float('inf')
            y_max = float('-inf')
            
            for line in ax.get_lines():
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                
                # PSCAD time data is sorted, so we can use O(log N) search
                idx_min = np.searchsorted(xdata, xmin, side='left')
                idx_max = np.searchsorted(xdata, xmax, side='right')
                
                if idx_min < idx_max:
                    visible_y = ydata[idx_min:idx_max]
                    y_min = min(y_min, float(np.nanmin(visible_y)))
                    y_max = max(y_max, float(np.nanmax(visible_y)))
                    
            if y_min != float('inf') and y_max != float('-inf'):
                margin = (y_max - y_min) * 0.05
                if margin == 0:
                    margin = 1.0
                ax.set_ylim(y_min - margin, y_max + margin)
                self._canvas.draw_idle()

if __name__ == "__main__":
    app = PSCADViewer()
    app.mainloop()
