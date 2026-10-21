# DCQSS-ATS: Code and Datasets for Audio Copy-Move Forgery Detection

Derived copy-move forgery datasets from Chinspeech, THCHS-30, TIMIT, and CHiME-6. Each dataset contains `untampered/` and `tampered/` speech audio files.

This repository also provides the Python DCQSS-ATS implementation used for the supplementary experiments, its exact signal-processing settings, confidence intervals, and paired significance tests. The numerical results below correspond to this implementation on the released WAV files with lambda=2.

## Environment and installation

Use Python 3.10.18 and the dependency versions in [requirements.txt](requirements.txt).

```bash
git lfs install
git clone https://github.com/acd26/DCQSS-ATS.git
cd DCQSS-ATS
git lfs pull
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. SoundFile requires libsndfile; if the installed wheel does not provide it, install the platform's libsndfile package. Git LFS must download actual WAV contents rather than leave pointer files.

## Repository files

| Path | Contents |
| --- | --- |
| `datasets/<dataset>/untampered/` | 500 authentic recordings per dataset |
| `datasets/<dataset>/tampered/` | 500 copy-move recordings per dataset |
| `code/dcqss.py` | Waveform preprocessing, YAAPT segmentation, CQSS, first differences, DCQSS, MSE and ATS |
| `code/cqt_backend.py` | Forward, fully rasterized real CQT backend |
| `code/evaluate.py` | End-to-end WAV evaluation for all three feature variants |
| `code/statistical_analysis.py` | Metric recomputation, recording-group bootstrap and paired exact McNemar tests |

## Exact method parameters

### Audio preprocessing and filtering

- Read audio with SoundFile at its stored sample rate; no external resampling is applied.
- Average channels for multichannel input, subtract the whole-recording mean, and divide by the whole-recording absolute peak if nonzero.
- No additional pre-emphasis is applied: there is no pre-emphasis coefficient in this implementation.
- YAAPT uses a 150th-order FIR band-pass filter (151 taps), 50–1500 Hz, designed by `scipy.signal.firwin(..., pass_zero=False)` with its Hamming-window default and applied using `scipy.signal.lfilter`. The decimation factor is 1. The squared-signal branch is also band-pass filtered inside pYAAPT.

### YAAPT segmentation

Voiced segments are contiguous runs of nonzero pitch. Analysis frames are 25 ms, the time-domain-analysis window is 35 ms, and the frame shift is 3 ms. FFT length is 8192 for Chinspeech and 2048 for THCHS-30, TIMIT and CHiME-6.

Runs of at most 8 pitch frames are marked short. Trailing short runs are removed; retained short runs use zero feature placeholders and are excluded from detection candidates.

The full YAAPT parameter dictionary and segment extraction are defined in `code/dcqss.py`.

### CQT and features

| Setting | Value |
| --- | --- |
| Bins per octave | 60 |
| Minimum frequency | 50 Hz |
| Maximum frequency | min(8000 Hz, sample rate/2) |
| Bandwidth offset | gamma = 228.7 × (2^(1/60) − 2^(−1/60)) |
| Analysis windows | Hann; toolbox-style DC/Nyquist boundary windows |
| Rasterization | Full; coefficient count of the highest positive-frequency channel |
| Normalization | Toolbox-style `sine`, factor 2 × coefficient count / segment sample count |
| Signal transform | Real, positive-frequency reduced form |
| CQSS | Temporal mean of log(abs(CQT)^2 + 2.2204e−16) |
| Difference variant | Prepend and append zero to CQSS, take first differences, append one final zero |
| DCQSS | Apply floor to the same difference vector |

### Similarity and threshold

For each pair, MSE is averaged over the smaller of the two feature-vector lengths. The symmetric distance matrix includes zero self-distances. Let I* be the median of its column medians; ATS uses `threshold = I*/lambda`, with lambda=2 in the fixed-protocol results below. Detection compares the minimum MSE among distinct, non-short segment pairs with the threshold, using `<=`.

When no such candidate pair exists, the detector returns no forgery detection.

## Run detection and statistical analysis

From the repository root:

```bash
# Full evaluation: all 4,000 WAVs, three feature variants
python code/evaluate.py --out outputs/decisions.csv

# Compute intervals and tests from locally generated decisions
python code/statistical_analysis.py --out outputs/statistics

```

The commands generate detection and statistical results locally under `outputs/`. Feature variants are `cqss` (original), `diff` (first difference without floor), and `dcqss` (first difference with floor).

## Statistical uncertainty

Positive means copy-move forgery. All metric values below are percentages.

The 95% confidence intervals use 2,000 paired bootstrap replicates with seed 777 and the 2.5th and 97.5th percentiles. Each authentic recording is resampled together with its forged counterpart. Within-session dependence is not captured by this source-pair resampling.

| Dataset | Metric | Estimate | 95% confidence interval |
| --- | --- | ---: | --- |
| Chinspeech | Precision | 99.38 | [98.56, 100.00] |
| Chinspeech | Recall | 95.80 | [94.00, 97.40] |
| Chinspeech | F1 | 97.56 | [96.52, 98.48] |
| Chinspeech | Accuracy | 97.60 | [96.60, 98.50] |
| THCHS-30 | Precision | 98.97 | [97.97, 99.79] |
| THCHS-30 | Recall | 95.80 | [94.00, 97.40] |
| THCHS-30 | F1 | 97.36 | [96.32, 98.37] |
| THCHS-30 | Accuracy | 97.40 | [96.40, 98.40] |
| TIMIT | Precision | 99.79 | [99.35, 100.00] |
| TIMIT | Recall | 93.60 | [91.40, 95.60] |
| TIMIT | F1 | 96.59 | [95.31, 97.65] |
| TIMIT | Accuracy | 96.70 | [95.50, 97.70] |
| CHiME-6 | Precision | 84.08 | [81.17, 87.25] |
| CHiME-6 | Recall | 63.40 | [59.00, 67.60] |
| CHiME-6 | F1 | 72.29 | [69.18, 75.28] |
| CHiME-6 | Accuracy | 75.70 | [73.50, 77.90] |

These are recording-level detection metrics, not segment-localization success rates. Running the statistical-analysis command above generates full-precision estimates and interval endpoints locally.

## Paired significance tests

Features are compared on the same recordings using the same segmentation, MSE, ATS and lambda=2. Wins/losses count recordings where only DCQSS/the comparator is correct. P-values use the two-sided exact McNemar test, without multiple-comparison or session-cluster adjustment.

| Dataset | Comparison | Wins | Losses | Exact p-value |
| --- | --- | ---: | ---: | ---: |
| Chinspeech | DCQSS vs CQSS | 472 | 20 | 3.1347e−113 |
| Chinspeech | DCQSS vs difference without floor | 21 | 12 | 0.16276 |
| THCHS-30 | DCQSS vs CQSS | 495 | 21 | 1.1654e−118 |
| THCHS-30 | DCQSS vs difference without floor | 147 | 16 | 1.0627e−27 |
| TIMIT | DCQSS vs CQSS | 330 | 31 | 2.9079e−64 |
| TIMIT | DCQSS vs difference without floor | 4 | 16 | 0.011818 |
| CHiME-6 | DCQSS vs CQSS | 435 | 178 | 8.7409e−26 |
| CHiME-6 | DCQSS vs difference without floor | 76 | 52 | 0.041638 |

Third-party dependencies remain under their respective licenses. The supplied code has no dependency on private workspace paths, cached feature files, pretrained weights or MATLAB at runtime.
