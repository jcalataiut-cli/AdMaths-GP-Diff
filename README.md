# AdMaths-GP-Diff

**GP-Diff: Gaussian Process Guided Diffusion Models for Time Series Forecasting**

## Structure

```
AdMaths-GP-Diff/
├── gp_diff/              # Core implementation
│   ├── __init__.py
│   ├── kernels.py        # GP-CPS kernel (SE, Periodic, Linear, Matern)
│   ├── gp_prior.py       # GP posterior computation
│   ├── diffusion.py      # Forward/reverse process with GP prior
│   ├── correction_net.py # Lightweight correction network δ_θ
│   ├── trainer.py        # Training pipeline
│   └── sampler.py        # Sampling pipeline
├── experiments/          # Jupyter notebooks with experiments
│   └── 01_gp_diff_demo.ipynb
├── figures/              # Generated figures for paper
├── paper/
│   ├── en/               # English version (IEEE)
│   └── es/               # Spanish version
├── requirements.txt
└── README.md
```
