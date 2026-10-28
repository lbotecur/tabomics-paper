# Results

- `raw/`: one row per fold, as written by the runs. In `raw/timing/`, `peak_gpu_mb` is the peak GPU memory allocated
  through PyTorch: it does not cover XGBoost (0 in its rows) or the CPU methods, which the paper reports as not
  recorded.
- `summaries/`: means per task and per benchmark, and statistical tests.
