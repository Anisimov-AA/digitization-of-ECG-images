# digitization-of-ECG-images

## Pipeline

1. **Stage 0** — Detect keypoints, apply homography to normalize image geometry
2. **Stage 1** — Detect ECG grid, rectify image to regular layout
3. **Stage 2** — Segment ECG traces with ensemble of whole + lead models
4. **Post-process** — Convert pixel masks to mV values via weighted average + resampling
5. **Submit** — Split 4 rows into 12 leads, format CSV

## Structure

```
ecg-digitization/
├── configs/
│   └── default.yaml           # paths, model list, hyperparams
├── src/
│   ├── __init__.py
│   ├── utils.py               # config loading, metadata helpers
│   ├── stage0.py              # keypoint detection → normalization
│   ├── stage1.py              # grid detection → rectification
│   ├── stage2.py              # segmentation → signal extraction
│   ├── postprocess.py         # pixel→series, resampling
│   ├── submission.py          # build submission CSV
│   └── metrics.py             # SNR computation for validation
├── notebooks/
│   └── inference.ipynb        # thin notebook for Kaggle submission
└── README.md
```

## Kaggle Datasets Required

- `hengck23-demo-submit-physionet` — Stage 0/1 models + code
- `physionet-final-submission-models` — Stage 2 models + code
- `my-pip-packages` — offline pip wheels (cc3d, smp)
