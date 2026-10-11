"""
Latency Limbo - the Streamlit app.

Catch falling fruits with your hand while a pose model tracks you through the
webcam. Pick a bigger model or a bigger input size and you will feel the lag.
Run it with: streamlit run app.py
"""

import math
import sys
import time

import cv2
import pandas as pd
import streamlit as st

from game import FruitGame
from detector import (MODELS, IMG_SIZES, DEFAULT_IMGSZ, DEFAULT_THREADS, HandTracker,
                      load_model, mps_available, prepare_frame, letterbox_shape,
                      count_params, weights_path)

COUNTDOWN_S = 3
JPEG_QUALITY = 80
STATS_EVERY = 0.25   # seconds between stats panel refreshes
PREVIEW_EVERY = 1.0  # the "what the model sees" image doesn't need to be live
MAX_DT = 0.1         # so a slow frame doesn't teleport the fruits
FAIR_SEED = 42
MAX_BAD_READS = 30

CAMERA_HELP = (
    "Could not open the webcam. On macOS, allow the app you started Streamlit from "
    "(Terminal, iTerm or VS Code) in System Settings > Privacy & Security > Camera, "
    "then restart it. Also close Zoom, FaceTime or other browser tabs that might be "
    "using the camera, or try another camera index under Advanced in the sidebar."
)


@st.cache_resource(show_spinner=False)
def get_model(model_file, device):
    # cached so we only load (and warm up) each model once per device
    return load_model(model_file, device)


def open_camera(index):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # the mac camera gives empty frames for a bit after opening, so keep trying
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def to_jpeg(frame):
    # st.image on a numpy array encodes at quality 100 which is slow, this is way faster
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def model_view(frame, imgsz, width=320):
    """Shrink the frame like the model does, then blow it back up so you can see the pixels."""
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    out_h = round(width * h / w)
    return cv2.resize(small, (width, out_h), interpolation=cv2.INTER_NEAREST)


def draw_big_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    (tw, th), _ = cv2.getTextSize(text, font, scale, 6)
    h, w = frame.shape[:2]
    org = ((w - tw) // 2, (h + th) // 2)
    # black outline first so the white text is readable on any background
    cv2.putText(frame, text, org, font, scale, (0, 0, 0), 6)
    cv2.putText(frame, text, org, font, scale, (255, 255, 255), 2)
    return frame


st.set_page_config(page_title="Latency Limbo", layout="wide")
st.title("Latency Limbo")
st.write("A webcam game where you catch fruits with your hand, and the hand tracker "
         "is the thing you are really testing. Bigger models see better but react later.")
st.markdown(
    "**How to play**\n"
    "1. Pick a model and an input size in the sidebar.\n"
    "2. Press Start round and sit back a bit so your shoulders and elbows are visible.\n"
    "3. Raise a hand. The white circle is where the model thinks your hand is, catch the fruits with it.\n"
    "4. Try other settings and compare the results below."
)

if "history" not in st.session_state:
    st.session_state["history"] = []

# ---- sidebar ----
st.sidebar.header("Settings")
label = st.sidebar.selectbox("Hand detector model", list(MODELS),
                             help="Tiny is fast but less accurate. Big is accurate but slow.")
imgsz = st.sidebar.selectbox("Input resolution", IMG_SIZES, index=IMG_SIZES.index(DEFAULT_IMGSZ),
                             format_func=lambda s: f"{s} px",
                             help="The camera frame is shrunk to this size before the model sees it.")
smoothing = st.sidebar.slider("Smoothing", 0.0, 0.9, 0.0, step=0.1,
                              help="Averages the hand position over time: less jitter, more lag.")
duration = st.sidebar.slider("Round length (seconds)", 15, 60, 30, step=5)
same_fruits = st.sidebar.checkbox("Same fruits every round", value=True,
                                  help="Uses the same random seed so rounds are a fair comparison.")
use_mps = st.sidebar.checkbox("Use Apple GPU (MPS)", value=False, disabled=not mps_available())
with st.sidebar.expander("Advanced"):
    cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1,
                                help="Try 1 if the wrong camera opens (for example an iPhone "
                                     "through Continuity Camera).")
    threads = st.slider("CPU threads for the model", 1, 8, DEFAULT_THREADS)
st.sidebar.caption("Changing a setting stops the current round.")

col_start, col_stop = st.columns(2)
start = col_start.button("Start round", type="primary")
stop = col_stop.button("Stop round")

video_col, stats_col = st.columns([3, 1])
video_slot = video_col.empty()
stats_slot = stats_col.empty()
preview_slot = stats_col.empty()

if stop:
    st.info("Round stopped. Unfinished rounds are not saved.")

if not start:
    video_slot.info("Press Start round. Your webcam turns on only during a round.")
else:
    # a thread from an older run can still be alive (threads survive reruns)
    old = st.session_state.pop("tracker", None)
    if old is not None:
        old.stop()

    model_file = MODELS[label]
    device = "mps" if use_mps else "cpu"
    with st.spinner(f"Loading {label} (the first time also downloads the weights)..."):
        model, device_used = get_model(model_file, device)
    if use_mps and device_used != "mps":
        st.warning("Apple GPU did not work here, using the CPU instead.")

    params_m = count_params(model) / 1e6
    file_mb = weights_path(model_file).stat().st_size / 1e6
    if device_used == "cpu":
        device_text = f"CPU, {threads} threads"
    else:
        device_text = "Apple GPU"

    cap = open_camera(int(cam_index))
    if cap is None:
        st.error(CAMERA_HELP)
        st.stop()

    tracker = HandTracker(model, imgsz=imgsz, device=device_used, smoothing=smoothing,
                          num_threads=threads)
    st.session_state["tracker"] = tracker
    tracker.start()

    finished = False
    error = None
    frame = None
    game = None
    round_start = None
    game_frames = 0
    final_stats = None
    round_end = None
    # careful: no st calls in the finally below, a rerun or Stop lands as an exception
    # at the next st call and we only want the cleanup to run
    try:
        ok, raw = cap.read()
        if not ok or raw is None:
            error = "Lost the camera feed."
        else:
            frame = prepare_frame(raw)
            h, w = frame.shape[:2]
            lh, lw = letterbox_shape(h, w, imgsz)
            seed = FAIR_SEED if same_fruits else None
            game = FruitGame(w, h, duration=duration, seed=seed)

            countdown_end = time.perf_counter() + COUNTDOWN_S
            last = time.perf_counter()
            last_stats = 0.0
            last_preview = 0.0
            fps_count = 0
            fps_start = last
            game_fps = 0.0
            bad_reads = 0
            playing = False

            while True:
                ok, raw = cap.read()
                if not ok or raw is None:
                    bad_reads += 1
                    if bad_reads >= MAX_BAD_READS:
                        error = "Lost the camera feed."
                        break
                    continue
                bad_reads = 0

                frame = prepare_frame(raw)
                now = time.perf_counter()
                tracker.submit(frame, now)
                state = tracker.latest()
                if state.error:
                    error = f"The detector crashed: {state.error}"
                    break
                age_ms = (now - state.frame_time) * 1000 if state.frame_time else None

                # show the shrunk input before we draw anything on the frame
                if now - last_preview >= PREVIEW_EVERY:
                    preview_slot.image(model_view(frame, imgsz), channels="BGR",
                                       caption=f"What the model sees ({lw}x{lh} px)")
                    last_preview = now

                left = countdown_end - now
                if left > 0:
                    game.draw(frame, state.hand, age_ms)
                    draw_big_text(frame, f"Get ready... {math.ceil(left)}")
                else:
                    if not playing:
                        # countdown frames shouldn't count in the stats
                        tracker.reset_stats()
                        round_start = now
                        game_frames = 0
                        playing = True
                    game.update(min(now - last, MAX_DT), state.hand)
                    game.draw(frame, state.hand, age_ms)
                    game_frames += 1
                last = now

                if smoothing > 0 and state.raw_hand is not None:
                    # yellow dot = raw detection, circle = smoothed one
                    rx, ry = int(state.raw_hand[0]), int(state.raw_hand[1])
                    cv2.circle(frame, (rx, ry), 4, (0, 255, 255), -1)

                video_slot.image(to_jpeg(frame), output_format="JPEG", width="stretch")

                fps_count += 1
                if now - fps_start >= 0.5:
                    game_fps = fps_count / (now - fps_start)
                    fps_count = 0
                    fps_start = now

                if now - last_stats >= STATS_EVERY:
                    s = tracker.summary()
                    lines = [
                        f"Model: {label} ({params_m:.1f} M params, {file_mb:.1f} MB)",
                        f"Input: {imgsz} px (model sees {lw}x{lh})",
                        f"Device: {device_text}",
                        f"Detector FPS: {state.detector_fps:.1f}",
                        f"Game FPS: {game_fps:.1f}",
                        f"Inference: {state.inference_ms:.0f} ms",
                        f"Delay camera to basket: {state.latency_ms:.0f} ms",
                        f"Frames skipped: {s['frames_skipped_pct']:.0f} %",
                        f"Caught / Missed: {game.caught} / {game.missed}",
                    ]
                    stats_slot.markdown("  \n".join(lines))
                    last_stats = now

                if game.over:
                    # grab the stats now, stop() below waits for the last inference
                    final_stats = tracker.summary()
                    round_end = now
                    finished = True
                    break
    finally:
        tracker.stop()
        cap.release()

    if finished:
        s = final_stats
        draw_big_text(frame, "Time's up!")
        video_slot.image(to_jpeg(frame), output_format="JPEG", width="stretch")

        history = st.session_state["history"]
        n = len(history) + 1
        short = label.split()[0]
        dev = "CPU" if device_used == "cpu" else "MPS"
        rate = game.catch_rate() * 100
        played = round_end - round_start
        history.append({
            "Run": f"#{n:02d} {short} {imgsz}px {dev}",
            "Model": short,
            "Input (px)": imgsz,
            "Device": dev,
            "Threads": threads,
            "Smoothing": round(smoothing, 1),
            "Caught": game.caught,
            "Missed": game.missed,
            "Catch rate (%)": round(rate, 1),
            "Detector FPS": round(s["detector_fps"], 1),
            "Inference (ms)": round(s["avg_inference_ms"], 1),
            "Delay (ms)": round(s["avg_latency_ms"], 1),
            "Hand found (%)": round(s["hand_found_pct"], 1),
            "Frames skipped (%)": round(s["frames_skipped_pct"], 1),
            "Game FPS": round(game_frames / played, 1) if played > 0 else 0.0,
        })
        c, m = game.caught, game.missed
        st.success(f"Round over! You caught {c} of {c + m} fruits ({rate:.0f}%).")
    if error:
        st.error(error)

st.divider()
st.subheader("Results")
history = st.session_state["history"]
if not history:
    st.caption("Play a round to see results here.")
else:
    if st.button("Clear results"):
        st.session_state["history"] = []
        st.rerun()
    df = pd.DataFrame(history)
    st.dataframe(df.drop(columns="Run"), hide_index=True)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.bar_chart(df, x="Run", y="Catch rate (%)", sort=False)
    with chart_right:
        st.bar_chart(df, x="Run", y="Detector FPS", sort=False)
    st.download_button("Download results (CSV)", df.to_csv(index=False),
                       "latency_limbo_results.csv", "text/csv", on_click="ignore")

with st.expander("What is going on here?"):
    st.markdown(
        "- The pose model finds 17 body points per person. We take your wrist and push it "
        "a bit further along the forearm so the circle sits on your palm.\n"
        "- The model runs in its own thread. While it is busy, new camera frames are skipped, "
        "so the video stays live but the circle only moves when the model finishes a frame.\n"
        "- A bigger model has more parameters and needs more compute, so each frame takes more ms. "
        "That means fewer detections per second, and the circle shows where your hand WAS.\n"
        "- Higher input resolution means more pixels to process. It is slower but the model "
        "sees more detail, which helps when you are far from the camera.\n"
        "- Smoothing averages the position over time. It removes jitter but adds even more lag.\n"
        "- The \"ms old\" label under the circle is how old the camera frame behind it is.\n"
        "- Edge devices like phones, Raspberry Pi, Jetson boards and smart cameras have little "
        "compute, power and memory, so a small fast model often beats a big accurate one in real time.\n"
        "- The Apple GPU option shows that hardware matters too: same model, very different speed."
    )

st.caption("Made by Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma and Hiten Mohindra.")
