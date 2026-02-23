import os
os.makedirs("outputs", exist_ok=True)

INPUT = "1006427285-0001.png"

steps = [
    "grid_scale.py",        # -> px_per_mm
    "remove_grid.py",      
    "remove_text.py",      
    "find_rows.py",         # -> bands, left, right
    "find_pulse.py",        # -> baselines, signal_starts, pulse_ranges
    "remove_artifacts.py",  # removes pulse/bar
    "find_leads.py",        # split by white gaps
    "trace.py",             # trace signal per lead, convert to mV
    "export_and_compare.py",
]

for step in steps:
    print(f">>> {step}")
    exec(open(step).read())