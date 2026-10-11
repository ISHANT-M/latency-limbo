import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from detector import (GREEN, L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, HandTracker, find_green_dot,
                      find_hand, hand_from_result, letterbox_shape, prepare_frame, smooth_point)


def person(lw=None, rw=None, le=None, re=None):
    """One fake person, keypoints given as (x, y) and anything missing gets conf 0."""
    xy = np.zeros((17, 2))
    conf = np.zeros(17)
    for idx, p in ((L_WRIST, lw), (R_WRIST, rw), (L_ELBOW, le), (R_ELBOW, re)):
        if p is not None:
            xy[idx] = p
            conf[idx] = 0.9
    return xy, conf


# find_hand

def test_find_hand_empty():
    assert find_hand(np.zeros((0, 17, 2)), np.zeros((0, 17))) is None


def test_find_hand_pushes_wrist_towards_palm():
    xy, conf = person(lw=(300, 200), le=(300, 300))
    x, y = find_hand(xy, conf)
    assert x == pytest.approx(300)
    assert y == pytest.approx(165)


def test_find_hand_low_conf_elbow_uses_wrist():
    xy, conf = person(lw=(300, 200), le=(300, 300))
    conf[L_ELBOW] = 0.1
    assert find_hand(xy, conf) == pytest.approx((300, 200))


def test_find_hand_both_wrists_low():
    xy, conf = person(lw=(300, 200), rw=(400, 250))
    conf[L_WRIST] = conf[R_WRIST] = 0.2
    assert find_hand(xy, conf) is None


def test_find_hand_picks_higher_hand():
    xy, conf = person(lw=(200, 300), rw=(400, 100))
    assert find_hand(xy, conf) == pytest.approx((400, 100))


def test_find_hand_sticks_to_previous_hand():
    xy, conf = person(lw=(200, 300), rw=(400, 100))
    assert find_hand(xy, conf, prev=(210, 290)) == pytest.approx((200, 300))


def test_find_hand_uses_biggest_person():
    small_xy, small_conf = person(lw=(100, 100))
    big_xy, big_conf = person(lw=(500, 300))
    xy = np.stack([small_xy, big_xy])
    conf = np.stack([small_conf, big_conf])
    boxes = np.array([[50, 50, 150, 200], [300, 0, 640, 480]])
    assert find_hand(xy, conf, boxes) == pytest.approx((500, 300))


def test_find_hand_clips_to_frame():
    # wrist near the top, elbow below it, so the palm point goes above the frame
    xy, conf = person(lw=(300, 10), le=(300, 200))
    x, y = find_hand(xy, conf, frame_shape=(480, 640, 3))
    assert y == 0.0
    assert x == pytest.approx(300)


def test_find_hand_no_conf_means_trust_everything():
    xy, _ = person(lw=(300, 200), le=(300, 300), rw=(400, 400), re=(400, 300))
    assert find_hand(xy, None) == pytest.approx((300, 165))
    # same answer with an explicit batch of one person
    assert find_hand(xy[None], None) == pytest.approx((300, 165))


# smooth_point

def test_smooth_point():
    assert smooth_point((0, 0), None, 0.5) is None
    assert smooth_point(None, (10, 20), 0.5) == (10, 20)
    assert smooth_point((0, 0), (10, 20), 0.0) == pytest.approx((10, 20))
    assert smooth_point((0, 0), (10, 20), 0.5) == pytest.approx((5, 10))
    # alpha gets clamped to 0.95 so it never freezes completely
    assert smooth_point((0, 0), (100, 100), 1.0) == pytest.approx((5, 5))


# frame helpers

def test_letterbox_shape():
    assert letterbox_shape(480, 640, 320) == (256, 320)
    assert letterbox_shape(480, 640, 640) == (480, 640)
    assert letterbox_shape(480, 640, 160) == (128, 160)
    assert letterbox_shape(360, 640, 640) == (384, 640)


def test_prepare_frame_resizes_and_mirrors():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :100] = 255  # white strip on the left
    out = prepare_frame(frame)
    assert out.shape == (360, 640, 3)
    assert out[:, -10:].mean() > 200  # strip is now on the right
    assert out[:, :10].mean() < 10

    no_flip = prepare_frame(frame, mirror=False)
    assert no_flip[:, :10].mean() > 200


def test_hand_from_result_with_fake():
    xy, conf = person(lw=(300, 200), le=(300, 300))
    fake = SimpleNamespace(
        keypoints=SimpleNamespace(xy=xy[None], conf=conf[None]),
        boxes=SimpleNamespace(xyxy=np.array([[200, 100, 400, 470]])),
        orig_shape=(480, 640),
    )
    assert hand_from_result(fake) == pytest.approx((300, 165))

    nothing = SimpleNamespace(keypoints=None, boxes=None, orig_shape=(480, 640))
    assert hand_from_result(nothing) is None


# HandTracker with a fake model

class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.processed = []

    def predict(self, frame, **kw):
        if self.fail:
            raise RuntimeError("boom")
        time.sleep(0.05)
        self.processed.append(int(frame[0, 0, 0]))
        xy, conf = person(lw=(300, 200))
        return [SimpleNamespace(
            keypoints=SimpleNamespace(xy=xy[None], conf=conf[None]),
            boxes=None, orig_shape=frame.shape[:2],
        )]


def wait_for(cond, timeout=3.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_tracker_skips_old_frames():
    model = FakeModel()
    tracker = HandTracker(model, num_threads=None)
    tracker.start()
    try:
        for v in range(1, 11):
            tracker.submit(np.full((480, 640, 3), v, dtype=np.uint8), time.perf_counter())
            time.sleep(0.005)
        assert wait_for(lambda: model.processed and model.processed[-1] == 10)
        time.sleep(0.1)
    finally:
        tracker.stop()

    assert len(model.processed) < 10
    assert all(a < b for a, b in zip(model.processed, model.processed[1:]))
    assert model.processed[-1] == 10
    state = tracker.latest()
    assert state.result_id >= 1
    assert state.raw_hand == pytest.approx((300, 200))
    s = tracker.summary()
    assert s["frames_skipped_pct"] > 0
    assert s["n_results"] == len(model.processed)
    assert not tracker.is_running


def test_tracker_stop_without_start():
    tracker = HandTracker(FakeModel(), num_threads=None)
    tracker.stop()
    tracker.stop()
    assert not tracker.is_running
    assert tracker.summary()["n_results"] == 0


def test_tracker_reports_errors():
    tracker = HandTracker(FakeModel(fail=True), num_threads=None)
    tracker.start()
    try:
        tracker.submit(np.zeros((48, 64, 3), dtype=np.uint8), time.perf_counter())
        assert wait_for(lambda: tracker.latest().error is not None)
        assert "RuntimeError" in tracker.latest().error
        assert wait_for(lambda: not tracker.is_running)
    finally:
        tracker.stop()


# real model

@pytest.mark.slow
def test_real_model_finds_a_hand():
    from ultralytics.utils import ASSETS
    from detector import load_model

    try:
        model, device = load_model("yolo11n-pose.pt", "cpu", warmup_sizes=[])
    except Exception as e:
        pytest.skip(f"could not load weights: {e}")

    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    assert img.shape[:2] == (720, 1280)
    res = model.predict(img, imgsz=640, device="cpu", verbose=False)
    hand = hand_from_result(res[0])
    assert hand is not None
    x, y = hand
    assert 640 < x < 1280
    assert 0 <= y < 720


# green sticker mode

def green_frame(center=(450, 150), radius=20, noise=False):
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)  # plain grey
    if center is not None:
        cv2.circle(frame, center, radius, (40, 200, 40), -1)  # BGR green
    if noise:
        # a few single green pixels, like camera noise
        for x, y in ((50, 50), (300, 400), (600, 20)):
            frame[y, x] = (40, 200, 40)
    return frame


@pytest.mark.parametrize("imgsz", [80, 160, 640])
def test_find_green_dot_centroid(imgsz):
    hand = find_green_dot(green_frame(), imgsz)
    assert hand is not None
    x, y = hand
    assert abs(x - 450) < 4
    assert abs(y - 150) < 4


def test_find_green_dot_no_green():
    assert find_green_dot(green_frame(center=None), 160) is None


def test_find_green_dot_ignores_tiny_noise():
    frame = green_frame(center=(320, 240), radius=2, noise=True)
    assert find_green_dot(frame, 640) is None


def test_find_green_dot_picks_biggest_blob():
    frame = green_frame(center=(100, 100), radius=10)
    cv2.circle(frame, (500, 300), 30, (40, 200, 40), -1)
    x, y = find_green_dot(frame, 320)
    assert abs(x - 500) < 4 and abs(y - 300) < 4


def test_tracker_green_mode():
    tracker = HandTracker(GREEN, imgsz=80)
    tracker.start()
    try:
        tracker.submit(green_frame(), time.perf_counter())
        assert wait_for(lambda: tracker.latest().result_id >= 1)
    finally:
        tracker.stop()
    state = tracker.latest()
    assert state.error is None
    assert state.raw_hand is not None
    assert abs(state.raw_hand[0] - 450) < 4
    assert tracker.summary()["hand_found_pct"] == 100.0
