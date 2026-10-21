"""Evaluate released WAVs without accessing any external workspace files."""

import argparse
from pathlib import Path
import pandas as pd
from dcqss import detect


def run(job):
    root, path, dataset, label, lam = job
    base = dict(
        dataset=dataset,
        label=label,
        stem=path.stem,
        path=path.relative_to(root).as_posix(),
    )
    try:
        return [dict(**base, status="ok", **r) for r in detect(path, dataset, lam)]
    except Exception as exc:
        return [
            dict(**base, status="error", error=type(exc).__name__ + ": " + str(exc))
        ]


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=["Chinspeech", "THCHS-30", "TIMIT", "CHiME-6"]
    )
    p.add_argument("--lambda-value", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("outputs/decisions.csv"))
    a = p.parse_args()
    if a.lambda_value <= 0:
        p.error("lambda-value must be positive")
    jobs = []
    for ds in a.datasets:
        for label in ["untampered", "tampered"]:
            files = sorted((root / "datasets" / ds / label).glob("*.wav"))
            if not files:
                raise FileNotFoundError(
                    f"No WAVs: {ds}/{label}; check Git LFS checkout"
                )
            jobs += [(root, path, ds, label, a.lambda_value) for path in files]
    rows = [r for job in jobs for r in run(job)]
    frame = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    if frame.status.ne("ok").any():
        raise RuntimeError(
            "Detection failures recorded in output; do not silently exclude them"
        )
    print(f"{len(jobs)} recordings; {len(frame)} feature/recording results: {a.out}")


if __name__ == "__main__":
    main()
