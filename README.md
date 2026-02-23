# digitization-of-ECG-images

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

All intermediate results saved to `outputs/` folder

## Pipeline

| Step | File                    | What it does                                     |
| ---- | ----------------------- | ------------------------------------------------ |
| 1    | `grid_scale.py`         | Measure grid spacing (px/mm) — grid is our ruler |
| 2    | `remove_grid.py`        | Remove red grid (R-G thresholding)               |
| 3    | `remove_text.py`        | Remove text labels (connected components)        |
| 4    | `find_rows.py`          | Find 4 lead rows (horizontal projection)         |
| 5    | `find_pulse.py`         | Find calibration pulse → baseline per row        |
| 6    | `remove_artifacts.py`   | Remove pulse + separator bars (shape detection)  |
| 7    | `find_leads.py`         | Split into 13 leads (white gap detection)        |
| 8    | `trace.py`              | Trace signal (weighted center of mass) → mV      |
| 9    | `export_and_compare.py` | Compare with ground truth                        |
