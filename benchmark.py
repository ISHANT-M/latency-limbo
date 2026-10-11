"""
Offline benchmark for the pose models.

Same idea as the game but without the webcam: run every model at every input
size on one test image and time it. Gives us numbers we can put in the report.

    python benchmark.py            # CPU only
    python benchmark.py --mps      # CPU and Apple GPU
"""

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics.utils import ASSETS

from detector import (DEFAULT_THREADS, IMG_SIZES, MAX_PEOPLE, MODELS, PERSON_CONF,
                      count_params, hand_from_result, letterbox_shape, load_model,
                      model_gflops, mps_available, weights_path)


def parse_args():
    ap = argparse.ArgumentParser(description="Time each pose model at each input size.")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+", default=IMG_SIZES)
    ap.add_argument("--models", nargs="+", default=list(MODELS.values()))
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--mps", action="store_true", help="also run on the Apple GPU")
    ap.add_argument("--image", default=None, help="test image (default: ultralytics zidane.jpg)")
    ap.add_argument("--out", default="results/benchmark.csv")
    return ap.parse_args()


def label_for(model_file):
    for label, f in MODELS.items():
        if f == model_file:
            return label.split()[0]
    return model_file


def bench_one(model, device, imgsz, img, runs, warmup, threads):
    kw = dict(imgsz=imgsz, device=device, conf=PERSON_CONF, max_det=MAX_PEOPLE, verbose=False)
    for _ in range(warmup):
        model.predict(img, **kw)
    if device == "cpu":
        # ultralytics changes the thread count when it builds the predictor
        torch.set_num_threads(threads)

    times, speeds, found = [], [], 0
    for _ in range(runs):
        t0 = time.perf_counter()
        r = model.predict(img, **kw)[0]
        hand = hand_from_result(r)  # included because the game does it too
        times.append((time.perf_counter() - t0) * 1000)
        speeds.append((r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]))
        found += hand is not None
    return np.array(times), np.array(speeds).mean(axis=0), found


def make_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devices = list(df["device"].unique())
    fig, axes = plt.subplots(1, len(devices), figsize=(6 * len(devices), 4.5), squeeze=False)
    for ax, dev in zip(axes[0], devices):
        sub = df[df["device"] == dev]
        models = list(sub["model"].unique())
        sizes = sorted(sub["imgsz"].unique())
        width = 0.8 / len(sizes)
        x = np.arange(len(models))
        for i, s in enumerate(sizes):
            vals = [sub[(sub["model"] == m) & (sub["imgsz"] == s)]["mean_ms"].mean() for m in models]
            ax.bar(x + (i - (len(sizes) - 1) / 2) * width, vals, width, label=f"{s} px")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_yscale("log")
        ax.set_ylabel("time per frame (ms, log scale)")
        ax.set_title(dev.upper())
        ax.legend(title="input size")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    image_path = args.image or str(ASSETS / "zidane.jpg")
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    # same size as our webcam frames after prepare_frame
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    devices = ["cpu"]
    if args.mps:
        if mps_available():
            devices.append("mps")
        else:
            print("MPS not available, CPU only")

    print(f"torch {torch.__version__} | {platform.platform()} | {platform.processor()} | "
          f"CPU threads: {args.threads}")

    rows = []
    for device in devices:
        for model_file in args.models:
            model, used = load_model(model_file, device, warmup_sizes=[])
            if used != device:
                print(f"skipping {model_file} on {device}, it fell back to {used}")
                continue
            params_m = count_params(model) / 1e6
            file_mb = weights_path(model_file).stat().st_size / 1e6
            for imgsz in args.sizes:
                times, speed, found = bench_one(model, device, imgsz, img,
                                                args.runs, args.warmup, args.threads)
                nh, nw = letterbox_shape(480, 640, imgsz)
                rows.append({
                    "device": device,
                    "model": label_for(model_file),
                    "model_file": model_file,
                    "params_m": round(params_m, 2),
                    "file_mb": round(file_mb, 1),
                    "imgsz": imgsz,
                    "input_hw": f"{nh}x{nw}",
                    "gflops": round(model_gflops(model, imgsz), 1),
                    "mean_ms": round(times.mean(), 1),
                    "median_ms": round(float(np.median(times)), 1),
                    "p90_ms": round(float(np.percentile(times, 90)), 1),
                    "std_ms": round(times.std(), 1),
                    "fps": round(1000 / times.mean(), 1),
                    "pre_ms": round(speed[0], 1),
                    "infer_ms": round(speed[1], 1),
                    "post_ms": round(speed[2], 1),
                    "hand_found": f"{found}/{args.runs}",
                    "threads": args.threads if device == "cpu" else "",
                })
                print(f"  {device} {model_file} @{imgsz}: {times.mean():.1f} ms")
            del model

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    png = out.with_suffix(".png")
    make_plot(df, png)
    print(f"\nsaved {out} and {png}")


if __name__ == "__main__":
    main()
