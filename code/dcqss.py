"""DCQSS-ATS: waveform preprocessing, YAAPT, CQSS/DCQSS and median ATS."""

from pathlib import Path
import numpy as np
import soundfile as sf
from amfm_decompy import basic_tools, pYAAPT
from cqt_backend import forward_full_real

# All pYAAPT 1.0.12.2 settings are explicit, including its defaults.
YAAPT = dict(
    frame_length=25.0,
    tda_frame_length=35.0,
    frame_space=3.0,
    f0_min=60.0,
    f0_max=400.0,
    bp_forder=150,
    bp_low=50.0,
    bp_high=1500.0,
    nlfer_thresh1=0.75,
    nlfer_thresh2=0.7,
    shc_numharms=3,
    shc_window=40.0,
    shc_maxpeaks=4,
    shc_pwidth=50.0,
    shc_thresh1=5.0,
    shc_thresh2=1.25,
    f0_double=150.0,
    f0_half=150.0,
    dp5_k1=11.0,
    dec_factor=1,
    nccf_thresh1=0.25,
    nccf_thresh2=0.9,
    nccf_maxcands=3,
    nccf_pwidth=5,
    merit_boost=0.2,
    merit_pivot=0.99,
    merit_extra=0.4,
    median_value=7,
    dp_w1=0.15,
    dp_w2=0.5,
    dp_w3=0.1,
    dp_w4=0.9,
    spec_pitch_min_std=0.05,
)


def load_audio(path):
    x, sr = sf.read(str(path))
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if not len(x):
        raise ValueError("Empty waveform")
    x = x - x.mean()
    peak = np.max(abs(x))
    if peak > 0:
        x = x / peak
    return x, sr


def extract(path, dataset):
    x, sr = load_audio(path)
    fft_length = 8192 if dataset.lower() == "chinspeech" else 2048
    pitch = np.asarray(
        pYAAPT.yaapt(
            basic_tools.SignalObj(x, sr), fft_length=fft_length, **YAAPT
        ).samp_values
    ).reshape(-1)
    pad = np.r_[0.0, pitch, 0.0]
    starts = np.where((pad[:-1] == 0) & (pad[1:] != 0))[0]
    ends = np.where((pad[:-1] != 0) & (pad[1:] == 0))[0] - 1
    segments = []
    for a, b in zip(starts, ends):
        if b < a:
            continue
        lo = max(0, int(a * int(sr * 0.003) + int(sr * 0.025) / 2))
        hi = min(len(x), int(b * int(sr * 0.003) + int(sr * 0.025) / 2) + 1)
        segments.append((lo, hi, b - a + 1 <= 8))
    while segments and segments[-1][2]:
        segments.pop()
    result = {name: [] for name in ["cqss", "diff", "dcqss"]}
    result["short"] = np.array([s[2] for s in segments], bool)
    for lo, hi, short in segments:
        if short:
            for name in ["cqss", "diff", "dcqss"]:
                result[name].append(np.zeros(439))
        else:
            feat = np.log(abs(forward_full_real(x[lo:hi], sr)) ** 2 + 2.2204e-16).mean(
                axis=1
            )
            diff = np.r_[np.diff(np.r_[0.0, feat, 0.0]), 0.0]
            result["cqss"].append(feat)
            result["diff"].append(diff)
            result["dcqss"].append(np.floor(diff))
    return result


def decision(features, short, lambda_value=2.0):
    if lambda_value <= 0:
        raise ValueError("lambda must be positive")
    n = len(features)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            length = min(len(features[i]), len(features[j]))
            d[i, j] = d[j, i] = np.mean(
                (features[i][:length] - features[j][:length]) ** 2
            )
    valid = np.flatnonzero(~np.asarray(short, bool))
    scores = [d[i, j] for k, i in enumerate(valid) for j in valid[k + 1 :]]
    i_star = float(np.median(np.median(d, axis=0))) if n >= 2 else float("inf")
    minimum = float(min(scores, default=float("inf")))
    threshold = i_star / lambda_value
    return dict(
        i_star=i_star,
        min_score=minimum,
        threshold=threshold,
        pred=bool(scores and minimum <= threshold),
    )


def detect(path, dataset, lambda_value=2.0):
    z = extract(Path(path), dataset)
    return [
        dict(variant=name, **decision(z[name], z["short"], lambda_value))
        for name in ["cqss", "diff", "dcqss"]
    ]
