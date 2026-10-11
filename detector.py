"""
Hand detection for Latency Limbo.

We don't have a hand model, so we use a YOLO pose model (17 body keypoints)
and take the wrist, pushed a bit towards the palm. The model runs in its own
thread so the game and the video never wait for it. That's the whole point:
the video stays smooth but the hand position is only as fresh as the last
finished inference.

Run `python detector.py` once to download all the weights into models/.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS = {
    "Tiny (YOLO11n-pose)": "yolo11n-pose.pt",
    "Small (YOLO11s-pose)": "yolo11s-pose.pt",
    "Big (YOLO11x-pose)": "yolo11x-pose.pt",
}
IMG_SIZES = [160, 320, 640]

# no neural net at all, just finds a green sticker on the palm
GREEN_LABEL = "Green sticker (colour tracking)"
GREEN = "green"  # pass this as the model to HandTracker
GREEN_SIZES = [80, 160, 320, 640]
GREEN_DEFAULT_IMGSZ = 160
GREEN_LOW = (40, 70, 70)     # HSV, OpenCV hue goes 0-179
GREEN_HIGH = (85, 255, 255)
GREEN_MIN_AREA = 0.0005      # smallest blob we accept, as a fraction of the image area
DEFAULT_IMGSZ = 320
FRAME_WIDTH = 640
DEFAULT_THREADS = 4  # pretend we're on a 4-core edge board

# COCO keypoint indices
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10

KPT_CONF = 0.4      # below this we don't trust a keypoint
PERSON_CONF = 0.3
MAX_PEOPLE = 5
PALM_EXTEND = 0.35  # how far past the wrist (as a fraction of the forearm) the palm is


def mps_available():
    return torch.backends.mps.is_available()


def weights_path(model_file):
    return MODELS_DIR / model_file


def _warmup(model, device, sizes):
    # first few predicts are slow (graph setup, MPS kernels etc.) so get them out of the way
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for s in sizes:
        model.predict(dummy, imgsz=s, device=device, conf=PERSON_CONF,
                      max_det=MAX_PEOPLE, verbose=False)


def load_model(model_file, device="cpu", warmup_sizes=IMG_SIZES):
    """Load a pose model (downloads it into models/ the first time)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if device == "mps" and not mps_available():
        device = "cpu"

    path = str(weights_path(model_file))
    model = YOLO(path)
    if device == "cpu":
        _warmup(model, device, warmup_sizes)
        return model, device

    try:
        _warmup(model, device, warmup_sizes)
    except Exception as e:
        # some ops still don't work on MPS, just go back to the CPU
        print(f"{device} failed ({type(e).__name__}: {e}), falling back to cpu")
        model = YOLO(path)
        device = "cpu"
        _warmup(model, device, warmup_sizes)
    return model, device


def count_params(model):
    return sum(p.numel() for p in model.model.parameters())


def model_gflops(model, imgsz):
    try:
        from ultralytics.utils.torch_utils import get_flops
        return float(get_flops(model.model, imgsz))
    except Exception:
        return 0.0


def prepare_frame(frame, width=FRAME_WIDTH, mirror=True):
    """Mirror (so it feels like a mirror) and resize to a fixed width."""
    # flip before inference, otherwise the keypoints come out on the wrong side
    if mirror:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, new_h), interpolation=interp)


def letterbox_shape(h, w, imgsz, stride=32):
    """Size of the image the model actually gets (ultralytics pads to a multiple of 32)."""
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    nh = int(np.ceil(nh / stride) * stride)
    nw = int(np.ceil(nw / stride) * stride)
    return nh, nw


def find_hand(kpts_xy, kpts_conf=None, boxes_xyxy=None, prev=None,
              kpt_conf=KPT_CONF, frame_shape=None):
    """Pick one hand point from the pose output. Returns (x, y) or None."""
    xy = np.asarray(kpts_xy, dtype=float)
    if xy.ndim == 2:
        xy = xy[None]
    if xy.size == 0 or xy.shape[0] == 0:
        return None
    if kpts_conf is None:
        conf = np.ones(xy.shape[:2])
    else:
        conf = np.asarray(kpts_conf, dtype=float).reshape(xy.shape[:2])

    # if there are several people, the biggest box is probably the player
    person = 0
    if boxes_xyxy is not None:
        boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
        if len(boxes) == xy.shape[0]:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            person = int(np.argmax(areas))
    pts, c = xy[person], conf[person]

    candidates = []
    for wrist, elbow in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
        if c[wrist] < kpt_conf:
            continue
        p = pts[wrist].copy()
        if c[elbow] >= kpt_conf:
            # the wrist is not the palm, extend along the forearm a bit
            p = p + PALM_EXTEND * (p - pts[elbow])
        candidates.append(p)
    if not candidates:
        return None

    if prev is not None:
        # stick with whichever hand we had last time, stops it jumping between hands
        prev = np.asarray(prev, dtype=float)
        best = min(candidates, key=lambda p: float(np.sum((p - prev) ** 2)))
    else:
        best = min(candidates, key=lambda p: p[1])  # higher hand = smaller y

    x, y = float(best[0]), float(best[1])
    if frame_shape is not None:
        h, w = frame_shape[0], frame_shape[1]
        x = min(max(x, 0.0), float(w - 1))
        y = min(max(y, 0.0), float(h - 1))
    return (x, y)


def _to_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


def hand_from_result(result, prev=None, kpt_conf=KPT_CONF):
    """find_hand but straight from an ultralytics Results object."""
    kp = result.keypoints
    if kp is None:
        return None
    xy = _to_numpy(kp.xy)
    conf = _to_numpy(kp.conf) if kp.conf is not None else None
    boxes = _to_numpy(result.boxes.xyxy) if result.boxes is not None else None
    return find_hand(xy, conf, boxes, prev=prev, kpt_conf=kpt_conf,
                     frame_shape=result.orig_shape)


def find_green_dot(frame, imgsz, prev=None):
    """Find the biggest green blob. Returns (x, y) in full frame pixels or None.

    prev is not used, it's only there so the call looks like the YOLO path.
    """
    mask, scale = green_mask(frame, imgsz)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    min_area = GREEN_MIN_AREA * mask.shape[0] * mask.shape[1]
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None
    m = cv2.moments(best)
    if m["m00"] == 0:
        return None
    # centroid from the moments, then back to full size coords
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return (float((cx + 0.5) / scale - 0.5), float((cy + 0.5) / scale - 0.5))


def green_mask(frame, imgsz):
    """Binary mask of the green pixels at the small size. Also returns the scale used."""
    h, w = frame.shape[:2]
    scale = min(1.0, imgsz / max(h, w))
    if scale < 1.0:
        small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame
    small = cv2.GaussianBlur(small, (3, 3), 0)  # kills a bit of camera noise
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # real scale can differ a tiny bit from imgsz/max because of rounding
    return mask, small.shape[1] / w


def smooth_point(prev, new, alpha):
    """Exponential smoothing. alpha 0 = no smoothing, higher = less jitter but more lag."""
    alpha = min(max(alpha, 0.0), 0.95)
    if new is None:
        return None
    if prev is None:
        return new
    return (alpha * prev[0] + (1 - alpha) * new[0],
            alpha * prev[1] + (1 - alpha) * new[1])


@dataclass
class TrackerState:
    hand: tuple = None          # smoothed hand position
    raw_hand: tuple = None      # what the model said, before smoothing
    frame_time: float = None    # perf_counter() of when the frame was captured
    inference_ms: float = 0.0
    latency_ms: float = 0.0     # capture -> result ready
    detector_fps: float = 0.0
    result_id: int = 0
    error: str = None


class HandTracker:
    """Runs the model on the newest frame in a background thread.

    If a new frame comes in while the model is busy, the older waiting frame is
    thrown away. So we always work on the freshest frame but skip a lot of them
    with a slow model.
    """

    def __init__(self, model, imgsz=DEFAULT_IMGSZ, device="cpu", smoothing=0.0,
                 num_threads=DEFAULT_THREADS, kpt_conf=KPT_CONF):
        self.model = model
        self.imgsz = imgsz
        self.device = device
        self.smoothing = smoothing
        self.num_threads = num_threads
        self.kpt_conf = kpt_conf

        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = None

        self._pending = None
        self._pending_time = None
        self._state = TrackerState()
        self._last_raw = None
        self._smoothed = None
        self._result_times = deque(maxlen=10)

        self.reset_stats()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def submit(self, frame, frame_time):
        with self._lock:
            if self._pending is not None:
                self.dropped += 1
            self._pending = frame.copy()
            self._pending_time = frame_time
            self.submitted += 1
        self._new_frame.set()

    def latest(self):
        with self._lock:
            return replace(self._state)

    def reset_stats(self):
        with self._lock:
            self.submitted = 0
            self.dropped = 0
            self.n_results = 0
            self.n_found = 0
            self.total_inference_ms = 0.0
            self.total_latency_ms = 0.0
            self.stats_start = time.perf_counter()

    def summary(self):
        with self._lock:
            n = self.n_results
            elapsed = time.perf_counter() - self.stats_start
            if n == 0:
                return {
                    "detector_fps": 0.0, "avg_inference_ms": 0.0, "avg_latency_ms": 0.0,
                    "hand_found_pct": 0.0,
                    "frames_skipped_pct": 100.0 * self.dropped / self.submitted if self.submitted else 0.0,
                    "n_results": 0,
                }
            return {
                "detector_fps": n / elapsed if elapsed > 0 else 0.0,
                "avg_inference_ms": self.total_inference_ms / n,
                "avg_latency_ms": self.total_latency_ms / n,
                "hand_found_pct": 100.0 * self.n_found / n,
                "frames_skipped_pct": 100.0 * self.dropped / self.submitted if self.submitted else 0.0,
                "n_results": n,
            }

    def stop(self, timeout=5.0):
        self._stop.set()
        self._new_frame.set()  # wake the thread up if it's waiting
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self):
        while not self._stop.is_set():
            if not self._new_frame.wait(timeout=0.1):
                continue
            with self._lock:
                frame, frame_time = self._pending, self._pending_time
                self._pending = None
                self._new_frame.clear()
            if frame is None:
                continue

            # ultralytics resets this when it builds the predictor, so set it back every time
            green = isinstance(self.model, str) and self.model == GREEN
            if not green and self.device == "cpu" and self.num_threads \
                    and torch.get_num_threads() != self.num_threads:
                torch.set_num_threads(self.num_threads)

            try:
                t0 = time.perf_counter()
                if green:
                    raw = find_green_dot(frame, self.imgsz, prev=self._last_raw)
                else:
                    res = self.model.predict(frame, imgsz=self.imgsz, device=self.device,
                                             conf=PERSON_CONF, max_det=MAX_PEOPLE, verbose=False)
                    raw = hand_from_result(res[0], prev=self._last_raw, kpt_conf=self.kpt_conf)
                t1 = time.perf_counter()
            except Exception as e:
                with self._lock:
                    self._state.error = f"{type(e).__name__}: {e}"
                break

            self._last_raw = raw
            self._smoothed = smooth_point(self._smoothed, raw, self.smoothing)
            self._result_times.append(t1)
            if len(self._result_times) > 1:
                span = self._result_times[-1] - self._result_times[0]
                fps = (len(self._result_times) - 1) / span if span > 0 else 0.0
            else:
                fps = 0.0

            inference_ms = (t1 - t0) * 1000
            latency_ms = (t1 - frame_time) * 1000
            with self._lock:
                self._state = TrackerState(
                    hand=self._smoothed, raw_hand=raw, frame_time=frame_time,
                    inference_ms=inference_ms, latency_ms=latency_ms,
                    detector_fps=fps, result_id=self._state.result_id + 1,
                )
                self.n_results += 1
                if raw is not None:
                    self.n_found += 1
                self.total_inference_ms += inference_ms
                self.total_latency_ms += latency_ms


if __name__ == "__main__":
    for label, file in MODELS.items():
        t = time.perf_counter()
        model, _ = load_model(file, "cpu", warmup_sizes=[320])
        took = time.perf_counter() - t
        mb = weights_path(file).stat().st_size / 1e6
        print(f"{label:22s} {file:18s} {mb:6.1f} MB  {count_params(model) / 1e6:5.1f} M params  "
              f"loaded in {took:.1f} s")
