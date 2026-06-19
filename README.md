# PSCAD Waveform Viewer

A Python-based waveform viewer for PSCAD output files (`.out` and `.inf`).

## Features

- Reads PSCAD `.out` and `.inf` files into a Pandas DataFrame.
- Loads `.out` files in parallel using a background thread.
- Automatically saves parsed data to a `.csv` cache file in the source directory for faster subsequent loads.
- Search bar to filter the list of available signals.
- Add, remove, and label multiple subplots.
- Aligns Y-axis labels vertically across subplots.
- Press `y` while hovering over a subplot to autoscale its Y-axis to the currently visible X-axis limits.

## Requirements

The application requires Python 3 and the following packages:

```bash
pip install pandas numpy matplotlib
```

## Usage

1. Run the script from the command line:
   ```bash
   python pscad_viewer.py
   ```
2. Click **Open Folder…** and select your PSCAD `.if18` output folder.
3. The signals will appear in the right-hand panel. You can use the search bar to filter the list.
4. Double-click a signal or select it and click **Add to subplot →** to plot it.
5. Use the dropdown menu to select the active subplot, or click **Add Subplot** to create a new one.
6. Use the Matplotlib toolbar at the bottom to zoom, pan, and save images.
7. To autoscale the Y-axis for a specific subplot based on the current X-axis zoom level, hover your mouse over the subplot and press the `y` key.
