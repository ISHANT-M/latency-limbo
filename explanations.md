# Latency Limbo - Explanations

Team: Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma, Hiten Mohindra

This document explains our project from top to bottom: what it does, why we built it this way, every important concept behind it, how each file works, what our measurements say, and the questions we expect in the viva. If you read it end to end you should be able to explain any line of the code.

---

## 1. How to run it

Setup (once):

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python detector.py
```

`python detector.py` loads all three pose models once. This downloads the weights into `models/` (about 145 MB in total) and prints the file size, parameter count and load time of each model. It is optional, the app downloads a model the first time you pick it, but then the first round takes longer to start.

Play the game:

```
streamlit run app.py
```

A browser tab opens. Pick a model and input size in the sidebar, press "Start round", sit back so your shoulders and elbows are visible, raise a hand and catch the fruits with the white circle.

Run the offline benchmark:

```
python benchmark.py            # CPU only
python benchmark.py --mps      # CPU and Apple GPU
python benchmark.py --help     # other options (--runs, --warmup, --sizes, --models, --threads, --image, --out)
```

It writes `results/benchmark.csv` and `results/benchmark.png`.

Run the tests:

```
pytest                  # everything
pytest -m "not slow"    # skip the test that loads real weights
```

Camera problems on macOS: the app you launched Streamlit from (Terminal, iTerm, VS Code) needs camera permission in System Settings > Privacy & Security > Camera. If the wrong camera opens (for example an iPhone through Continuity Camera), change "Camera index" under Advanced in the sidebar.

---

## 2. The big idea

In most ML courses we compare models by accuracy. On an edge device that is only half the story. A model that is very accurate but takes 234 ms per frame is describing where your hand was a quarter of a second ago. In a real time system that old answer can be worse than a slightly less accurate but fresh answer.

We wanted people to feel this, not just read it in a table. So we made a game:

- Fruits fall from the top of the webcam image.
- A pose model finds your hand, and a white circle (the "basket") is drawn there.
- If the basket touches a fruit, you catch it.

With the Tiny model the circle follows your hand closely. With the Big model at 640 px on the CPU the circle lags behind and jumps, and you miss fruits even though the model "sees" you very well. The game turns latency into a score.

Three lessons:

1. Bigger models are usually more accurate but slower, and slow means stale.
2. Input resolution is a separate knob: fewer pixels is faster, but the model sees less detail.
3. Hardware matters: the same model on the Apple GPU (MPS) runs much faster than on 4 CPU threads.

We also wrote an offline benchmark so we have repeatable numbers to back up what you feel in the game.

---

## 3. Concepts

### 3.1 Latency vs throughput

- Latency: how long one item takes from start to finish. For us, from the moment a camera frame is captured to the moment the hand position from that frame is ready. Measured in ms.
- Throughput: how many items are finished per second. For us, detector FPS.

For a single model working on one frame at a time, throughput is roughly 1000 / latency. Tiny@320 on CPU takes 8.8 ms, so about 114 FPS. But they are not the same thing in general. With batching or pipelining you can raise throughput without lowering latency (you finish more frames per second, but each one still takes as long). For an interactive game, latency is what the player feels.

### 3.2 The different FPS numbers

We show three different "FPS" ideas and it is easy to mix them up:

- Camera FPS: how many frames the webcam delivers per second. A laptop webcam gives about 30 FPS. This is an upper limit on everything else in the live game.
- Game FPS: how many times per second our main loop reads a frame, updates the game and draws it. In `app.py` it is counted over 0.5 s windows (`game_fps`). It is mostly tied to the camera, because `cap.read()` waits for the next frame.
- Detector FPS: how many inference results per second the background thread produces. Live, this comes from `TrackerState.detector_fps`, computed over the last 10 results. In the round summary it is `n_results / elapsed` from `HandTracker.summary()`.
- Benchmark FPS: the `fps` column in the CSV, which is just `1000 / mean_ms`. This is the speed the model could reach if frames were always ready instantly.

Important: in the live game, detector FPS cannot go above camera FPS, because the detector only runs when a new frame arrives. So even though Tiny@320 can do 114 FPS in the benchmark, in the game it is capped around the camera rate (~30 FPS). In a test with a fake camera, Tiny@320 ran at about 26 detector FPS with about 13 ms delay. A slow model like Big@640 on CPU (about 4.3 FPS) is far below the camera rate, so it skips most frames.

### 3.3 Where the delay comes from

The "Delay camera to basket" number is `latency_ms = (t1 - frame_time) * 1000` in `HandTracker._run`:

- `frame_time` is `time.perf_counter()` taken right after `cap.read()` returns the frame in `app.py`.
- `t1` is taken after `model.predict(...)` and `hand_from_result(...)` finish.

So the measured delay is:

1. Waiting time in the mailbox: the frame sits in `_pending` until the worker thread is free. If the model is busy with the previous frame, this can be almost a full inference time.
2. Preprocessing: resize, letterbox, convert to a tensor (`pre_ms` in the benchmark, about 0.2 to 0.5 ms).
3. Inference: the network itself (`infer_ms`, the big part).
4. Postprocessing: non max suppression and scaling keypoints back to the frame (`post_ms`).
5. Our `find_hand` code (tiny).

On top of what we measure there are delays we do not measure: the camera sensor exposure and driver buffer before `cap.read()` returns, and the time to JPEG encode the frame, send it to the browser and show it on the screen. So the real "glass to glass" delay the player feels is larger than our number. We say this honestly in the limitations.

There is also the "ms old" label under the circle, `age_ms = (now - state.frame_time) * 1000`. That is the age of the frame behind the circle at the moment it is drawn. It keeps growing between results, so with a slow model it climbs to a few hundred ms and then drops when a new result arrives.

### 3.4 Input resolution and letterbox

Our camera frames are 640x480 (width x height) after `prepare_frame`. The model does not take that directly. Ultralytics resizes the frame so the longer side equals `imgsz`, keeping the aspect ratio, and then pads the short side up to a multiple of 32 (the network's stride). This is called letterboxing. Our function `letterbox_shape(h, w, imgsz, stride=32)` reproduces this:

```
r = min(imgsz / h, imgsz / w)
nh, nw = round(h * r), round(w * r)
round both up to a multiple of 32
```

Real shapes for our 480x640 frames (height x width), also checked in `tests/test_detector.py` and in the `input_hw` column of the CSV:

| imgsz | after resize | after padding (what the model sees) |
|---|---|---|
| 160 | 120x160 | 128x160 |
| 320 | 240x320 | 256x320 |
| 640 | 480x640 | 480x640 (already multiples of 32) |

A 16:9 frame of 360x640 at imgsz 640 becomes 384x640 (360 padded up to 384).

Why multiples of 32? YOLO downsamples the image by 2 five times (strides 8, 16, 32 are used for the detection heads), so the size must divide evenly by 32 or feature maps would not line up.

The app shows "What the model sees" in the sidebar column. `model_view` in `app.py` shrinks the frame the same way the model does (INTER_AREA) and then blows it back up to 320 px wide with INTER_NEAREST so you can see the blocky pixels. At 160 px your hand is only a few pixels wide.

### 3.5 FLOPs grow with pixels

A convolution does the same work at every position of the feature map. So compute grows roughly with the number of pixels. Doubling the width and height gives 4x the pixels and about 4x the FLOPs. Our CSV shows this clearly (GFLOPs from `model_gflops`):

| model | 160 | 320 | 640 |
|---|---|---|---|
| Tiny | 0.5 | 1.9 | 7.6 |
| Small | 1.5 | 5.8 | 23.4 |
| Big | 12.7 | 51.0 | 204.2 |

From 320 to 640 the pixel count goes 256x320 = 81,920 to 480x640 = 307,200, which is 3.75x, and Tiny's GFLOPs go 1.9 to 7.6 (4x). Time does not scale perfectly with FLOPs: small inputs have fixed overheads (Python, framework, memory), and the CPU is used more efficiently on bigger tensors. For example Big goes from 67.2 ms to 233.6 ms (3.5x) for 4x FLOPs, and Tiny goes from 4.3 ms at 160 to 8.8 ms at 320 (2x) for about 4x FLOPs.

### 3.6 Parameters vs FLOPs vs file size

These are three different measures of "how big" a model is:

- Parameters: the number of learned weights. Tiny 2.87 M, Small 9.92 M, Big 58.8 M (from `count_params`).
- File size: how much disk space the weights take. Tiny 6.3 MB, Small 20.4 MB, Big 118.5 MB. Roughly params x 2 bytes, because the `.pt` files store weights in half precision (FP16) plus a bit of extra data. This matters for download size and memory on an edge device.
- FLOPs: how much computation one forward pass needs. This depends on the input size, the parameter count does not. Tiny at 640 (7.6 GFLOPs) needs less compute than Big at 160 (12.7 GFLOPs).

Parameters and file size tell you about memory and storage. FLOPs tell you about speed. Speed in practice also depends on memory bandwidth, how well the hardware is used, and overheads.

### 3.7 YOLO11 pose and the 17 COCO keypoints

We use three YOLO11 pose models from Ultralytics: `yolo11n-pose` (nano, we call it Tiny), `yolo11s-pose` (small) and `yolo11x-pose` (extra large, we call it Big). They are single stage models: one forward pass gives, for every person, a bounding box, a confidence score and 17 keypoints, each with (x, y) and its own confidence. They were trained on the COCO keypoints dataset.

The 17 COCO keypoints in order:

| index | point | index | point |
|---|---|---|---|
| 0 | nose | 9 | left wrist |
| 1 | left eye | 10 | right wrist |
| 2 | right eye | 11 | left hip |
| 3 | left ear | 12 | right hip |
| 4 | right ear | 13 | left knee |
| 5 | left shoulder | 14 | right knee |
| 6 | right shoulder | 15 | left ankle |
| 7 | left elbow | 16 | right ankle |
| 8 | right elbow | | |

In `detector.py`: `L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10`.

We call `model.predict(frame, imgsz=..., device=..., conf=PERSON_CONF, max_det=MAX_PEOPLE, verbose=False)` with `PERSON_CONF = 0.3` (ignore people found with less than 30 % confidence) and `MAX_PEOPLE = 5`.

Why pose and not a hand model? Ultralytics does not ship a pretrained hand detector, and a pose model gives us three sizes of the same family, so comparing them is fair. Pose models also need the body in view, which is why we ask players to sit back.

### 3.8 The wrist + elbow palm trick

COCO has a wrist point but no palm point. When you hold your hand up, the wrist is a few cm below where you would "catch" with your palm. So in `find_hand` we extend the forearm a little:

```
p = wrist + PALM_EXTEND * (wrist - elbow)     # PALM_EXTEND = 0.35
```

`wrist - elbow` is the direction from elbow to wrist, so we move 35 % of the forearm length further past the wrist. Example from our tests: wrist (300, 200), elbow (300, 300). Forearm vector is (0, -100), so p = (300, 200) + 0.35 x (0, -100) = (300, 165).

If the elbow confidence is below `KPT_CONF = 0.4` we just use the wrist. If the wrist itself is below 0.4 that hand is not a candidate at all. If no hand is a candidate, `find_hand` returns `None` and the game shows "no hand found - raise your hand!". The final point is clipped into the frame if `frame_shape` is given.

### 3.9 Which person and which hand

- Which person: if there are several people and we have their boxes, we take the person with the biggest box area. The player is usually closest to the camera so their box is biggest. Without boxes we take person 0.
- Which hand: from the candidate hands (left and/or right):
  - If we had a hand last time (`prev`, which is the previous raw result), we take the candidate closest to it. This stops the circle jumping between your two hands every frame.
  - Otherwise we take the higher hand (smaller y, because image y grows downwards). The raised hand is the one you are playing with.

### 3.10 EMA smoothing

Keypoints jitter a little from frame to frame. `smooth_point(prev, new, alpha)` does an exponential moving average:

```
smoothed = alpha * prev + (1 - alpha) * new
```

- alpha 0 = no smoothing (default in the app).
- Higher alpha = less jitter but more lag, because old positions count more. Roughly, the circle needs about 1 / (1 - alpha) results to catch up: at alpha 0.5 about 2 results, at 0.9 about 10 results. With a slow model each result is already old, so smoothing makes it much worse.
- alpha is clamped to at most 0.95 so the point never freezes completely.
- If the new point is `None`, the result is `None` (we do not keep showing a hand that has gone). If there is no previous point, we take the new one directly.

When smoothing is on, the app also draws a small yellow dot at the raw (unsmoothed) point so you can see the difference between the two.

### 3.11 Threading, the latest frame mailbox and frame dropping

The easy way to write this would be: read frame, run model, draw, repeat. Then with Big@640 on CPU the whole video would freeze at about 4 FPS. Instead we split the work:

- Main thread (in `app.py`): reads the camera, updates and draws the game, and sends images to the browser. It never waits for the model.
- Worker thread (`HandTracker._run`): runs the model again and again on the newest frame.

They talk through a "mailbox" that holds only one frame (`_pending`):

- `submit(frame, frame_time)`: under a lock, puts a copy of the frame in the mailbox. If there was already a frame waiting that the worker had not picked up, it is overwritten and `dropped` goes up by 1. Then it sets a `threading.Event` to wake the worker.
- `_run`: waits on the event (with a 0.1 s timeout so it can notice a stop request), takes the frame out of the mailbox under the lock, runs the model, and publishes a new `TrackerState` under the lock.
- `latest()`: returns a copy of the current state (`dataclasses.replace`) under the lock, so the main thread never sees a half written state.

Why drop frames instead of queueing them? A queue would make the model work through old frames one by one, and the delay would keep growing without limit. With a single slot mailbox the model always starts on the freshest frame, so the delay stays bounded at roughly one inference time plus waiting. The cost is that with a slow model most frames are skipped, which is the "Frames skipped" percentage (`100 * dropped / submitted`).

We copy the frame in `submit` because the main thread draws fruits on the same frame right after; without the copy the model could see our drawings.

Other details:

- The thread is a daemon thread named "hand-tracker", so it cannot keep Python alive by itself.
- `stop()` sets the stop event, also sets the new frame event to wake the worker, and joins with a 5 s timeout. It is safe to call twice or before `start()`.
- If `predict` raises an exception, the worker stores `"TypeName: message"` in `state.error` and exits. The app sees this and shows "The detector crashed: ...".
- Streamlit reruns the whole script on every widget change, but threads survive reruns. So we keep the tracker in `st.session_state` and stop any old one before starting a new round.

### 3.12 The GIL

Python's Global Interpreter Lock lets only one thread run Python bytecode at a time. So how do two threads help? Because most of the heavy work happens outside Python bytecode: PyTorch's convolution kernels, OpenCV's `cap.read()`, `cv2.resize` and JPEG encoding are C/C++ code that release the GIL while they run. So the model can compute while the main thread reads the camera. There is still some Python overhead in both threads competing for the GIL, which is one reason the live numbers are a bit worse than the benchmark. For full separation you would use separate processes, which costs more (frames have to be copied between processes).

### 3.13 CPU threads as an edge proxy

We do not have a Raspberry Pi or Jetson for the demo, so we use the laptop CPU limited to a few threads as a stand in for a small edge board. `DEFAULT_THREADS = 4` ("pretend we're on a 4-core edge board"). Before each inference on CPU, `_run` calls `torch.set_num_threads(self.num_threads)` if the current value differs, because Ultralytics resets the thread count when it builds its predictor. The benchmark does the same after warm-up. The thread count can be changed from 1 to 8 under Advanced.

This is a proxy, not a real measurement of an edge board. A real 4 core ARM board is slower per core than a laptop CPU, has less memory bandwidth and may throttle when hot. So the real numbers would be worse, but the trend (Tiny much faster than Big, 160 much faster than 640) stays the same.

### 3.14 MPS (Apple GPU)

MPS (Metal Performance Shaders) is PyTorch's backend for the GPU on Apple Silicon Macs. With "Use Apple GPU (MPS)" ticked, we pass `device="mps"`. `load_model` warms the model up on MPS, and if any operation fails it reloads the model on CPU and tells the user ("Apple GPU did not work here, using the CPU instead.").

What the numbers show:

- For Big@640 MPS takes 111.6 ms vs 233.6 ms on CPU (about 2.1x faster).
- For Tiny@320 they are about the same: 8.8 ms on CPU and 9.1 ms on MPS. The model is so small that the GPU's fixed costs (sending data, launching kernels, waiting for results) eat the gain. Tiny@160 is even slower on MPS (8.1 ms) than on CPU (4.3 ms).
- On MPS `post_ms` is large (for Big@640, 96.8 ms of postprocessing vs 7.6 ms "inference"). This is because the GPU runs asynchronously: `infer_ms` only measures queueing the work, and the real wait happens when results are copied back to the CPU during postprocessing. So on MPS only the total (`mean_ms`) is meaningful, not the split.

Lesson: an accelerator helps big models a lot and small models barely or not at all.

### 3.15 Edge constraints

Edge devices (phones, Raspberry Pi, Jetson boards, smart cameras) have:

- Little compute: a few CPU cores, maybe a small GPU or NPU.
- Little memory and storage: a 118.5 MB model may not fit well, a 6.3 MB one does.
- Power and heat limits: running flat out drains batteries and causes thermal throttling.
- Real time deadlines: a camera at 30 FPS gives about 33 ms per frame. Tiny@320 (8.8 ms) fits easily. Big@640 (234 ms) misses it by 7x.
- Often no network, or you do not want to send video to a server for privacy and latency reasons, which is why the model runs on the device at all.

So on the edge we choose the smallest model and input size that is accurate enough, not the most accurate model.

### 3.16 Making models faster (future work, we did not implement these)

- Quantization: store weights (and activations) in INT8 instead of FP32/FP16. About 4x smaller than FP32 and faster on hardware with INT8 support, with a small accuracy loss. Can be post training or quantization aware training.
- Pruning: remove weights or whole channels that contribute little, then fine tune. Structured (channel) pruning gives real speedups; unstructured pruning mostly needs special hardware to help.
- Knowledge distillation: train a small "student" model to copy the outputs of a big "teacher". The student can get closer to the teacher's accuracy than training alone.
- ONNX: an open model format. Exporting to ONNX lets us run with ONNX Runtime, which applies graph optimizations (operator fusion, constant folding) and supports many hardware backends.
- TFLite (LiteRT): TensorFlow's runtime for mobile and microcontrollers, with good INT8 support and delegates for phone GPUs/NPUs.

Ultralytics can export to ONNX, TFLite, CoreML, OpenVINO and others with `model.export(format=...)`, so these are realistic next steps.

---

### 3.17 Green sticker mode: classic colour tracking

This is the fourth option in the model picker. The player sticks a green dot (or a bit of green tape) on the palm, and instead of a neural network we find it with plain OpenCV. It is here as a comparison: what if we drop deep learning completely?

The steps in `find_green_dot(frame, imgsz)`:

1. Shrink the frame so the long side is `imgsz` (INTER_AREA). Same idea as the input size for YOLO, fewer pixels means less work. 80 px is also allowed here, because a big green blob is still visible at 80x60.
2. Small Gaussian blur (3x3) to remove a bit of camera noise.
3. Convert BGR to HSV.
4. `cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)` with `GREEN_LOW = (40, 70, 70)` and `GREEN_HIGH = (85, 255, 255)`. Every pixel inside the range becomes 255, everything else 0. This is the binary mask (the app shows it as "Green mask" in the preview).
5. Morphological open with a 3x3 kernel: erode then dilate. Erode removes single noisy pixels, dilate grows the real blob back to roughly its size.
6. `cv2.findContours` on the mask (outer contours only) and pick the biggest one by `cv2.contourArea`.
7. If its area is below `GREEN_MIN_AREA` (0.05 % of the image area, so it scales with imgsz) we say no hand. Otherwise take the centroid from image moments: `cx = m10 / m00`, `cy = m01 / m00`. `m00` is the area, `m10` and `m01` are the sums of x and y over the blob, so this is just the average position of the blob pixels.
8. Scale the centroid back to full frame pixels and return it.

Why HSV and not RGB: in RGB (BGR in OpenCV) the brightness is mixed into all three channels. The same green sticker in shadow might be (20, 80, 20) and in bright light (120, 250, 120), so a simple box in RGB space does not catch both. HSV splits colour into Hue (which colour, 0 to 179 in OpenCV, green is around 60), Saturation (how colourful vs grey) and Value (how bright). So we can say "hue between 40 and 85, not too grey, not too dark" and lighting changes mostly move V, not H. The minimum S of 70 stops white or grey walls from counting, the minimum V of 70 stops very dark pixels (where the hue is basically random) from counting.

Why it is so fast: there is no network at all. Every step is a simple operation per pixel (resize, blur, colour convert, compare, 3x3 erode/dilate) plus contour tracing, so the cost is roughly O(number of pixels) with tiny constants, and OpenCV does it in optimized C++ with SIMD. At 160 px that is only 19,200 pixels. Compare with YOLO, where even Tiny@160 needs about 0.5 GFLOPs. 0 params, 0 MB of weights, nothing to load or warm up, and it does not use PyTorch, so the CPU threads setting does nothing here.

Numbers from our benchmark (CPU): 0.22 ms at 80, 0.06 ms at 160, 0.16 ms at 320 and 0.52 ms at 640. Tiny@160, the fastest YOLO setting, is 4.3 ms, so even green@640 is about 8x faster, and green@160 is about 70x faster. One odd thing: 80 is slower than 160. That is OpenCV's INTER_AREA resize: it has a fast path when shrinking by 2x or 4x, but 640 to 80 is 8x and goes through the slower general code. We checked: resizing to 80x60 with INTER_AREA took about 0.2 ms, to 160x120 about 0.01 ms. INTER_LINEAR at 80 would be faster but skips pixels, so a small dot could flicker. In the game it does not matter, both are far below a millisecond.

In the live game it is capped by the camera just like YOLO: in our fake camera test green@80 gave about 25 detector FPS with about 1 ms delay, against about 26 FPS and 12 ms for Tiny@320. So the circle is almost exactly on the current frame.

Limitations (why everyone does not just do this):

- You need a marker. It tracks a green dot, not a hand. Without the sticker it finds nothing, or worse, finds something else.
- Any other green thing in view (a plant, a green t-shirt, a green wall) can be bigger than the sticker and wins. Even our test image has a greenish patch of about 940 px in the corner at 640, which beat a 15 px radius dot at 320 and 640, so the benchmark uses a 25 px radius dot.
- Lighting: very warm or dim light shifts the colours, the hue range may need tuning. Strong glare makes the sticker look white (low saturation). Auto white balance on webcams also moves colours around.
- Fixed thresholds. We picked the range by hand. A real system would calibrate at the start (for example click on the sticker and learn its colour).
- No understanding: it cannot tell a palm from a leaf. A pose model knows what a person looks like and works for any player with no setup.

Classical CV vs deep learning, and the edge angle: classical CV is hand designed rules (colour ranges, edges, shapes). It needs almost no compute and memory, is easy to explain, and has no training data, but it only works in the conditions it was designed for. Deep learning learns features from data, so it handles the messy real world (any person, clothes, background, light) but needs much more compute, memory and a big trained model. On an edge device this matters a lot: a microcontroller or a cheap camera can run colour tracking at full frame rate with almost no power, where even a tiny network may be too slow. In industry both are used, often together: if you control the scene (a factory line, a marker on a robot, a lab setup), classic CV is cheap and good enough. If you do not control the scene, you need a learned model. A common trick is a cheap classic step first (motion or colour) and only run the network when something interesting happens.

---

## 4. File by file walkthrough

### 4.1 `game.py` - game logic, no ML

This file knows nothing about neural networks. It gets a hand position `(x, y)` or `None` each frame. This is why we can test it without a webcam.

Constants:

- `FRUIT_TYPES`: apple, orange, lime, grape, each with a BGR colour (OpenCV uses BGR) and a radius (22, 22, 18, 15 px).
- `BASKET_RADIUS = 50`: how close the hand must be to catch.
- Difficulty: `SPAWN_START = 1.1` s between fruits at the start, `SPAWN_DROP = 0.5` (gap shrinks to 0.6 s by the end), `BASE_SPEED = 170` px/s, `SPEED_RAMP = 90` (extra speed by the end), `SPEED_SPREAD = 60` (random extra). We tuned these by playing.

`Fruit` is a dataclass with `x, y, speed, kind` and a `radius` property.

`FruitGame(width, height, duration=45, seed=None)`:

- Uses its own `random.Random(seed)`. Same seed means the exact same fruits, positions and speeds every round, so rounds with different models are a fair comparison.
- `time_to_next_spawn = 0.5`, so the first fruit appears after half a second.
- `over` is true when `elapsed >= duration`; `time_left` never goes below 0.
- `spawn_interval()`: `SPAWN_START - SPAWN_DROP * progress`, progress = elapsed / duration.
- `spawn_fruit()`: random kind, random x with a 40 px margin, starts at y = -20 (just above the screen), speed between `base` and `base + SPEED_SPREAD` where `base = BASE_SPEED + SPEED_RAMP * progress`.
- `update(dt, hand)`: does nothing if over. Adds dt to elapsed, spawns if the timer runs out, moves each fruit down by `speed * dt`. Catch test is circle vs circle: caught if the squared distance from fruit centre to hand is at most `(BASKET_RADIUS + fruit.radius) ** 2` (squared to avoid a square root). A fruit is missed when its top edge is below the bottom of the frame. Each catch or miss adds a popup ("+1" or "miss") that lives 0.6 s.
- `catch_rate()`: caught / (caught + missed), 0.0 if nothing happened yet.
- `draw(frame, hand, hand_age_ms=None)`: draws fruits (filled circle, dark outline, a little stem), the white basket circle and centre dot with the "ms old" label, or the "no hand found" message, the popups, and a black HUD bar with caught, missed and time left. Draws in place and returns the same frame.

### 4.2 `detector.py` - models, hand point, background thread

Constants (all explained above): `MODELS_DIR`, `MODELS` (label to weights file), `IMG_SIZES = [160, 320, 640]`, `DEFAULT_IMGSZ = 320`, `FRAME_WIDTH = 640`, `DEFAULT_THREADS = 4`, keypoint indices, `KPT_CONF = 0.4`, `PERSON_CONF = 0.3`, `MAX_PEOPLE = 5`, `PALM_EXTEND = 0.35`. For the green mode: `GREEN_LABEL`, `GREEN = "green"` (what we pass to `HandTracker` instead of a model), `GREEN_SIZES = [80, 160, 320, 640]`, `GREEN_DEFAULT_IMGSZ = 160`, `GREEN_LOW`, `GREEN_HIGH`, `GREEN_MIN_AREA`.

Functions:

- `mps_available()`: `torch.backends.mps.is_available()`.
- `weights_path(model_file)`: `models/<file>`.
- `_warmup(model, device, sizes)`: runs predict on a black 480x640 image once per size. The first predictions are slow (graph setup, memory allocation, MPS kernel compilation), and we do not want that inside a round.
- `load_model(model_file, device="cpu", warmup_sizes=IMG_SIZES)`: creates `models/`, falls back to CPU if MPS is not available, loads `YOLO(path)` (downloads if missing), warms up. On a non CPU device any exception during warm-up makes it reload on CPU. Returns `(model, device_used)`.
- `count_params(model)`: sums `numel()` over all parameters.
- `model_gflops(model, imgsz)`: uses Ultralytics' `get_flops`, returns 0.0 if that fails.
- `prepare_frame(frame, width=640, mirror=True)`: flips horizontally so it feels like a mirror, then resizes to 640 wide keeping the aspect ratio (INTER_AREA when shrinking, INTER_LINEAR when enlarging). We flip before inference, so keypoints and the drawn image are in the same coordinates. If we flipped only the display, the circle would be on the wrong side.
- `letterbox_shape(h, w, imgsz, stride=32)`: the shape the model really sees (section 3.4).
- `find_hand(...)`: pure numpy. Accepts (17, 2) for one person or (N, 17, 2). Picks the person, builds hand candidates, picks one, clips (sections 3.8 and 3.9).
- `_to_numpy(x)`: converts torch tensors (`.cpu().numpy()`) or arrays to float numpy.
- `hand_from_result(result, prev=None, kpt_conf=KPT_CONF)`: takes an Ultralytics `Results` object, pulls out `keypoints.xy`, `keypoints.conf`, `boxes.xyxy` and `orig_shape`, and calls `find_hand`. Written so it also works on fake objects in tests.
- `smooth_point(prev, new, alpha)`: EMA (section 3.10).
- `green_mask(frame, imgsz)`: shrink, blur, HSV, inRange, open. Returns the mask and the scale used. The app also uses it for the preview.
- `find_green_dot(frame, imgsz, prev=None)`: biggest contour above the min area, centroid from moments, scaled back to full frame coords (section 3.17). `prev` is not used, it just keeps the same shape as the YOLO call.

`TrackerState` dataclass: `hand` (smoothed), `raw_hand`, `frame_time`, `inference_ms`, `latency_ms`, `detector_fps`, `result_id` (counts results), `error`.

`HandTracker` (section 3.11): `start`, `is_running`, `submit`, `latest`, `reset_stats`, `summary`, `stop`, `_run`.

- Live detector FPS uses a `deque(maxlen=10)` of result times: `(len - 1) / (last - first)`. It reacts quickly to changes.
- `summary()` returns averages since the last `reset_stats()`: `detector_fps` (results / elapsed), `avg_inference_ms`, `avg_latency_ms`, `hand_found_pct`, `frames_skipped_pct`, `n_results`. The app calls `reset_stats()` when the countdown ends so the countdown frames do not count.
- Inference time is `t1 - t0` around predict plus `hand_from_result`; latency is `t1 - frame_time`.
- If the model is the string `GREEN`, `_run` calls `find_green_dot` instead of predict plus `hand_from_result`. Everything else (mailbox, smoothing, stats, timing) is the same code, so green numbers can be compared directly with the YOLO ones. `load_model` is never called for green mode.

`python detector.py` loads each model on CPU (warm-up only at 320) and prints its size, params and load time.

### 4.3 `app.py` - the Streamlit app

Constants: `COUNTDOWN_S = 3`, `JPEG_QUALITY = 80`, `STATS_EVERY = 0.25` s, `PREVIEW_EVERY = 1.0` s, `MAX_DT = 0.1` s, `FAIR_SEED = 42`, `MAX_BAD_READS = 30`, and a `CAMERA_HELP` message.

Helpers:

- `get_model(model_file, device)`: wrapped in `@st.cache_resource` so each model is loaded and warmed up only once per device, not on every rerun.
- `open_camera(index)`: uses the AVFoundation backend on macOS, asks for 640x480, then tries up to 30 reads 0.05 s apart because the Mac camera returns empty frames right after opening. Returns `None` if it fails.
- `to_jpeg(frame)`: encodes at quality 80. Passing raw numpy arrays to `st.image` encodes at quality 100, which was much slower.
- `model_view(frame, imgsz)`: the pixelated "What the model sees" preview.
- `draw_big_text(frame, text)`: centred text with a black outline, used for the countdown and "Time's up!".

Page layout: title, intro, "How to play"; sidebar "Settings" with model, input resolution, smoothing (0 to 0.9), round length (15 to 60 s, default 30), "Same fruits every round" (default on), "Use Apple GPU (MPS)" (disabled if not available), and under Advanced the camera index and CPU threads (1 to 8, default 4). Buttons "Start round" and "Stop round". A wide video column and a narrow stats column.

Round flow when "Start round" is pressed:

1. Stop any old tracker left in `st.session_state`.
2. Load the model (cached), warn if MPS fell back to CPU. Compute params and file size for the stats panel.
3. Open the camera or show the help message and stop.
4. Create `HandTracker(model, imgsz, device, smoothing, threads)`, save it in session state, start it.
5. Read the first frame, `prepare_frame` it, compute the letterbox shape, create `FruitGame(w, h, duration, seed=42 or None)`.
6. Main loop:
   - Read a frame. Empty reads are counted; after 30 in a row we stop with "Lost the camera feed."
   - `prepare_frame`, `now = perf_counter()`, `tracker.submit(frame, now)`, `state = tracker.latest()`.
   - If the detector crashed, stop with the error.
   - Once per second, update the model preview (before drawing on the frame).
   - During the 3 s countdown, draw the game and "Get ready... N", but do not update it.
   - After the countdown: the first time, `reset_stats()` and remember `round_start`. Then `game.update(min(now - last, MAX_DT), state.hand)` and `game.draw(...)`. `MAX_DT` stops fruits from teleporting if one loop iteration was slow.
   - If smoothing is on, draw the yellow raw dot.
   - Show the frame as JPEG.
   - Every 0.25 s, update the stats panel: model with params and MB, input size and letterbox shape, device, detector FPS, game FPS, inference ms, delay camera to basket, frames skipped %, caught / missed.
   - When `game.over`, take `tracker.summary()` and leave the loop.
7. In `finally`: stop the tracker and release the camera. There are no Streamlit calls inside `finally`, because when the user presses Stop or changes a setting, Streamlit interrupts the script with an exception at the next `st` call, and we only want cleanup there.
8. If the round finished, draw "Time's up!", append a row to `st.session_state["history"]` and show a success message. Unfinished (stopped) rounds are not saved.

Results section: a table of all finished rounds (model, input, device, threads, smoothing, caught, missed, catch rate, detector FPS, inference ms, delay ms, hand found %, frames skipped %, game FPS), two bar charts (catch rate and detector FPS per run), a "Clear results" button and a CSV download. There is also a "What is going on here?" expander with a short explanation for players.

### 4.4 `benchmark.py` - offline speed test

Why: live numbers depend on the camera, the browser and how you move. The benchmark gives clean, repeatable numbers.

- Image: Ultralytics' sample `zidane.jpg`, resized to 640x480 so it matches our webcam frames.
- For each device (CPU, plus MPS with `--mps`), each model and each size: load without warm-up, then `bench_one` does `--warmup` untimed predicts (default 5), sets the CPU thread count (default 4), then times `--runs` predicts (default 30). Each timed run includes `hand_from_result`, because the game does that too.
- It records mean, median, p90 and std of the times, `fps = 1000 / mean`, the Ultralytics pre/inference/post split, params, file size, GFLOPs, the letterbox shape and how many runs found a hand.
- Unless `--no-green` is given, it also adds 4 green sticker rows (CPU, sizes 80/160/320/640). It draws a green dot on the zidane frame at a known spot and counts a run as found only if the centroid is within 10 px of it. Pre/infer/post, GFLOPs and threads are empty or 0 for these rows.
- Saves the CSV and a plot: one subplot per device, grouped bars per model and size, log scale y axis (because values go from 4 ms to 234 ms).

Our saved CSV was made with 20 timed runs per setting (the `hand_found` column shows 20/20).

### 4.5 `tests/`

`pytest.ini` sets `pythonpath = .`, `testpaths = tests`, and a `slow` marker.

- `test_game.py`: hand on fruit catches it; `None` hand never catches; far hand does not catch; fruit below the bottom is a miss; the timer ends the game and nothing changes after; catch rate 0.0 and 0.5; first fruit after 0.5 s; same seed gives the same fruits, different seeds differ; `draw` keeps the frame shape and actually draws something.
- `test_detector.py`:
  - `find_hand`: empty input, the palm extension (300, 200) + elbow (300, 300) gives (300, 165), low confidence elbow falls back to the wrist, both wrists low gives `None`, picks the higher hand, sticks to the previous hand, picks the biggest person, clips to the frame, `None` confidence means trust everything.
  - `smooth_point` cases including the 0.95 clamp.
  - `letterbox_shape` with the four shapes from section 3.4.
  - `prepare_frame` turns 720x1280 into 360x640 and mirrors.
  - `hand_from_result` with a fake result made from `SimpleNamespace`.
  - `HandTracker` with a `FakeModel` whose predict sleeps 50 ms and records which frame it saw. We submit frames 1 to 10 every 5 ms. The test checks that fewer than 10 were processed (frames were dropped), they were processed in increasing order (never an old frame after a new one), the last one processed was 10 (the newest frame is never lost), and the frames skipped % is above 0. Other tests check stop without start and that a crashing model sets `error` and the thread exits.
  - Green mode: `find_green_dot` on a grey frame with a green circle gives the centre within 4 px at 80, 160 and 640, no green gives `None`, a few single green pixels and a tiny dot (below the min area) give `None`, two blobs picks the bigger one, and `HandTracker(GREEN, imgsz=80)` returns the hand with 100 % found.
  - A `slow` test loads the real Tiny model on `zidane.jpg` and checks that a hand is found in the right half of the image.
- `test_app.py`: uses Streamlit's `AppTest` to load the page without exceptions, check the title, check the model picker has three models plus the green sticker option, and check the empty results caption. It never presses Start, so the webcam is not opened.

---

## 5. Results

### 5.1 Benchmark results (from `results/benchmark.csv`)

Setup: 640x480 test image, CPU limited to 4 threads, and Apple GPU (MPS). mean_ms is the average time for predict plus hand extraction.

CPU, 4 threads:

| model | params (M) | file (MB) | imgsz | model input | GFLOPs | mean ms | FPS |
|---|---|---|---|---|---|---|---|
| Tiny | 2.87 | 6.3 | 160 | 128x160 | 0.5 | 4.3 | 230.3 |
| Tiny | 2.87 | 6.3 | 320 | 256x320 | 1.9 | 8.8 | 114.0 |
| Tiny | 2.87 | 6.3 | 640 | 480x640 | 7.6 | 24.8 | 40.3 |
| Small | 9.92 | 20.4 | 160 | 128x160 | 1.5 | 7.2 | 139.0 |
| Small | 9.92 | 20.4 | 320 | 256x320 | 5.8 | 15.6 | 64.1 |
| Small | 9.92 | 20.4 | 640 | 480x640 | 23.4 | 48.9 | 20.4 |
| Big | 58.8 | 118.5 | 160 | 128x160 | 12.7 | 27.8 | 36.0 |
| Big | 58.8 | 118.5 | 320 | 256x320 | 51 | 67.2 | 14.9 |
| Big | 58.8 | 118.5 | 640 | 480x640 | 204.2 | 233.6 | 4.3 |

MPS (Apple GPU):

| model | imgsz | mean ms | FPS |
|---|---|---|---|
| Tiny | 160 | 8.1 | 123.8 |
| Tiny | 320 | 9.1 | 109.5 |
| Tiny | 640 | 14.7 | 68.1 |
| Small | 160 | 10.1 | 98.5 |
| Small | 320 | 14.2 | 70.3 |
| Small | 640 | 24.6 | 40.6 |
| Big | 160 | 27.6 | 36.3 |
| Big | 320 | 44.8 | 22.3 |
| Big | 640 | 111.6 | 9.0 |

Green sticker mode (CPU, no network, test image is the same zidane frame with a green dot of radius 25 px drawn at (520, 120)):

| imgsz | image it works on | mean ms | median ms | FPS | found |
|---|---|---|---|---|---|
| 80 | 60x80 | 0.22 | 0.22 | 4475 | 20/20 |
| 160 | 120x160 | 0.06 | 0.06 | 16459 | 20/20 |
| 320 | 240x320 | 0.16 | 0.16 | 6438 | 20/20 |
| 640 | 480x640 | 0.52 | 0.51 | 1926 | 20/20 |

All runs found a hand on the test image (20/20), so on this easy image all models are "accurate enough". For the green rows "found" only counts if the dot found is within 10 px of where we drew it.

The chart is in `results/benchmark.png` (log scale).

### 5.2 How to read them

- Model size: at 320 on CPU, Tiny 8.8 ms, Small 15.6 ms, Big 67.2 ms. Big has about 20x the parameters of Tiny and is about 7.6x slower.
- Input size: Tiny goes 4.3, 8.8, 24.8 ms for 160, 320, 640. Big goes 27.8, 67.2, 233.6 ms.
- Real time budget: a 30 FPS camera gives about 33 ms per frame. On CPU, everything except Big@320, Big@640 and Small@640 fits in that (Big@160 at 27.8 ms barely fits). Big@640 at 234 ms is about 7 camera frames late.
- Hardware: MPS makes Big@640 go from 233.6 ms to 111.6 ms but does nothing for Tiny@320 (8.8 ms on CPU, 9.1 ms on MPS).
- Stability: std is small on CPU (at most 4.2 ms, for Big@640), MPS is a bit noisier (up to 9.3 ms for Big@640), and median is close to mean, so the measurements are steady.
- The `fps` column is the most the model could do. In the game the camera caps it at about 30 FPS (we saw about 26 detector FPS for Tiny@320 with about 13 ms delay in a fake camera test). For Tiny@320 the delay was about 13 ms vs 8.8 ms in the benchmark: the extra is time waiting in the mailbox and Python/GIL overhead in the live loop.
- MPS pre/infer/post split is not meaningful because GPU work is asynchronous (section 3.14). Only `mean_ms` counts.

- Green sticker: 0.06 to 0.52 ms with 0 params. That is about 8x (green@640) to 70x (green@160) faster than the fastest YOLO setting (Tiny@160, 4.3 ms), and about 450x to 3900x faster than Big@640 on CPU. But that only says it is fast on an image where we put a sticker, not that it tracks hands (section 3.17).

Our default is Tiny@320 on 4 CPU threads: fast enough for the camera, and accurate enough when you sit at a normal distance.

### 5.3 Our game rounds (fill in after playing)

Use "Same fruits every round" (seed 42), 30 s rounds, smoothing 0, same player, same lighting and distance. Copy the numbers from the Results table in the app or from the downloaded CSV.

| run | model | input (px) | device | threads | caught | missed | catch rate (%) | detector FPS | inference (ms) | delay (ms) | hand found (%) | frames skipped (%) | game FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Tiny | 320 | CPU | 4 | | | | | | | | | |
| 2 | Small | 320 | CPU | 4 | | | | | | | | | |
| 3 | Big | 320 | CPU | 4 | | | | | | | | | |
| 4 | Big | 640 | CPU | 4 | | | | | | | | | |
| 5 | Tiny | 160 | CPU | 4 | | | | | | | | | |
| 6 | Tiny | 640 | CPU | 4 | | | | | | | | | |
| 7 | Big | 640 | MPS | - | | | | | | | | | |
| 8 | Tiny | 320 | CPU | 4 (smoothing 0.7) | | | | | | | | | |

What we expect: catch rate falls and frames skipped rises as delay goes up. Tiny@160 may lose the hand more often (lower hand found %) when you are far from the camera. Smoothing should lower the catch rate a bit even though the circle looks calmer. Play each setting at least twice and report the average, since one round is noisy.

---

## 6. Viva questions and answers

**Q1. What does your project do in one sentence?**
It is a webcam game where you catch falling fruits with your hand, tracked by a YOLO11 pose model, so you can feel how a bigger or slower model makes the tracking lag and your score drop.

**Q2. Why is this an Edge AI project?**
The whole point is the trade off that matters on edge devices: limited compute means you must balance accuracy against latency. We run on the device (no server), limit the CPU to 4 threads to act like a small board, and measure latency, FPS, params, FLOPs and file size.

**Q3. What is the difference between latency and throughput?**
Latency is the time for one frame from capture to result (ms). Throughput is results per second (FPS). For one frame at a time, FPS is about 1000 / latency, but batching or pipelining can raise throughput without lowering latency. For a game, latency is what you feel.

**Q4. Your benchmark says Tiny@320 is 114 FPS. Why does the game show around 30?**
The detector only runs when the camera gives a new frame, and the webcam gives about 30 FPS. So detector FPS is capped by the camera. In a fake camera test we saw about 26 detector FPS for Tiny@320.

**Q5. What exactly is "Delay camera to basket"?**
`latency_ms = (t1 - frame_time) * 1000`. `frame_time` is when `cap.read()` returned the frame, `t1` is when predict and hand extraction finished. It includes waiting in the mailbox, pre-processing, inference and post-processing.

**Q6. What delay does it not include?**
Camera sensor and driver delay before `cap.read()` returns, and JPEG encoding, sending to the browser and screen refresh after. So the real delay felt by the player is larger.

**Q7. Why run the model in a separate thread?**
So the video and the game never wait for the model. With Big@640 on CPU (234 ms) a single loop would freeze the video at about 4 FPS. With a thread, the video stays at camera rate and only the circle updates slowly.

**Q8. What happens to frames that arrive while the model is busy?**
The mailbox holds only one frame. A new frame overwrites the waiting one and `dropped` goes up. The worker always takes the newest frame. This is counted as "Frames skipped".

**Q9. Why not use a queue so no frame is lost?**
If the model is slower than the camera, the queue grows forever and the delay grows with it. After 10 seconds you would be seeing results from many seconds ago. Dropping keeps delay bounded and the result as fresh as possible.

**Q10. Python has a GIL. How can two threads help?**
The GIL only blocks running Python bytecode. PyTorch kernels and OpenCV functions (camera read, resize, JPEG encode) are native code that release the GIL, so the model can compute while the main thread reads and draws. Some Python overhead still competes, which is one reason live numbers are a bit worse.

**Q11. How do you make the threads safe?**
A `threading.Lock` protects the mailbox, the state and the counters. `latest()` returns a copy of the state, so the main thread never reads a half updated object. A `threading.Event` wakes the worker when a frame arrives, and another event tells it to stop.

**Q12. Why do you copy the frame in submit?**
The main thread draws fruits and text on the frame right after submitting. Without a copy the model could see our drawings or a half drawn frame.

**Q13. How do you stop the thread cleanly?**
`stop()` sets the stop event, sets the new frame event to wake the worker if it is waiting, and joins with a 5 s timeout. The worker also waits with a 0.1 s timeout so it checks the stop flag often. It is a daemon thread, and `stop()` is safe to call twice or without `start()`. We call it in a `finally` block in the app.

**Q14. What if the model crashes inside the thread?**
The exception is caught, saved as `"TypeName: message"` in `state.error`, and the thread exits. The main loop checks `state.error` and shows "The detector crashed: ...". We test this with a fake model that raises.

**Q15. Why did you use a pose model instead of a hand detector?**
Ultralytics provides pretrained pose models in several sizes of the same family, which makes the comparison fair, but no hand model. The wrist keypoint, pushed towards the palm, works well enough for a game.

**Q16. What are the 17 COCO keypoints and which do you use?**
Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles (indices 0 to 16). We use left/right elbow (7, 8) and left/right wrist (9, 10).

**Q17. Explain the palm trick with numbers.**
`p = wrist + 0.35 * (wrist - elbow)`. Wrist (300, 200), elbow (300, 300): the forearm vector is (0, -100), so p = (300, 165). We move 35 % of the forearm length past the wrist, roughly where the palm is.

**Q18. What if the elbow is not visible?**
If the elbow confidence is below 0.4 we just use the wrist. If the wrist is below 0.4 that hand is ignored. If neither hand is usable we return `None` and show "no hand found".

**Q19. How do you choose between two people and two hands?**
Person: the biggest bounding box, since the player is usually closest. Hand: the one closest to last time's hand to avoid jumping; if there is no previous hand, the higher one (smaller y), since you raise the hand you play with.

**Q20. What is EMA smoothing and why is it off by default?**
`smoothed = alpha * prev + (1 - alpha) * new`. It reduces jitter but adds lag, about 1 / (1 - alpha) results to catch up. Our project is about latency, so we start with no extra lag and let the player try it.

**Q21. Why clamp alpha at 0.95?**
At alpha 1 the point would never move. Clamping keeps at least 5 % of each new result.

**Q22. What is letterboxing?**
Resizing so the longer side equals `imgsz` while keeping the aspect ratio, then padding the shorter side to a multiple of 32. Our 480x640 frame at imgsz 320 becomes 240x320 and is padded to 256x320.

**Q23. Why multiples of 32?**
The network downsamples by 2 five times, so the largest stride is 32. Sizes must divide evenly so the feature maps of different levels line up.

**Q24. How do FLOPs change with input size?**
Roughly with the number of pixels. Doubling width and height gives about 4x FLOPs: Tiny 1.9 to 7.6 GFLOPs, Big 51.0 to 204.2 GFLOPs from 320 to 640.

**Q25. Does time grow exactly like FLOPs?**
No. Big went from 67.2 to 233.6 ms (3.5x) for 4x FLOPs, and Tiny from 4.3 to 8.8 ms (2x) for about 4x FLOPs from 160 to 320. Small inputs have fixed overheads, and memory bandwidth and hardware usage also matter.

**Q26. Difference between params, FLOPs and file size?**
Params: number of learned weights (Tiny 2.87 M, Big 58.8 M). File size: storage (6.3 MB vs 118.5 MB, about 2 bytes per param since weights are stored in FP16). FLOPs: compute per forward pass, depends on input size. Params and size tell you memory needs, FLOPs tell you about speed.

**Q27. Can a smaller model at a bigger input be slower than a bigger model at a smaller input?**
In FLOPs, Tiny@640 (7.6) is still below Big@160 (12.7), and timing agrees (24.8 vs 27.8 ms). But Small@640 (23.4 GFLOPs, 48.9 ms) is slower than Big@160 (12.7 GFLOPs, 27.8 ms). So "small model" alone does not mean fast, input size counts as much.

**Q28. Why limit the CPU to 4 threads?**
To act like a 4 core edge board instead of a full laptop CPU. It is a proxy. A real board would be slower, but the trends would be the same.

**Q29. Why do you set the thread count before every inference?**
Ultralytics resets the PyTorch thread count when it builds its predictor, so we set it back if it changed. The benchmark does the same after warm-up.

**Q30. Why does MPS help Big but not Tiny?**
The GPU has fixed costs per call (moving data, launching kernels, syncing). For Big@640 the compute is large so MPS wins (111.6 vs 233.6 ms). For Tiny the compute is tiny, so the overhead dominates: 8.8 ms on CPU vs 9.1 ms on MPS at 320, and MPS is slower at 160 (8.1 vs 4.3 ms).

**Q31. Why is post_ms so big on MPS?**
GPU work is asynchronous. "Inference" only records sending the work. The actual waiting happens when results are copied back to the CPU during post-processing. So only the total is meaningful on MPS.

**Q32. What is warm-up and why do it?**
The first few predictions are slow because of setup (memory allocation, graph building, GPU kernel compilation). We run dummy predictions on a black image at load time, and the benchmark does 5 untimed runs, so these costs do not appear in measurements or in the game.

**Q33. How did you make comparisons between models fair in the game?**
"Same fruits every round" uses seed 42, so the same fruits fall at the same places and speeds. Stats are reset after the 3 s countdown. The same camera frames size (640x480) is used for all models.

**Q34. Why is dt capped at 0.1 s?**
If one loop iteration is slow (for example the first frame or a hiccup), a large dt would teleport fruits. Capping at `MAX_DT = 0.1` keeps movement smooth.

**Q35. How is a catch decided?**
Circle overlap: caught if the squared distance from the fruit centre to the hand is at most `(BASKET_RADIUS + fruit.radius) ** 2`. We compare squared values to avoid a square root.

**Q36. Why mirror the frame before inference and not just for display?**
If we only flipped the display, the keypoints (from the unflipped frame) would be on the wrong side of the screen. Flipping first keeps the model and the drawing in the same coordinates.

**Q37. How do you test the threading without a camera or real model?**
A `FakeModel` that sleeps 50 ms and records which frame it saw. We submit 10 frames 5 ms apart and check fewer than 10 were processed, they were in increasing order, the last one (10) was processed, and frames skipped is above 0.

**Q38. Why is game logic in a separate file?**
`game.py` has no ML. It only takes a hand position or `None`. So we can test it with exact positions and swap the detector without touching the game.

**Q39. What does the benchmark measure exactly?**
For each device, model and size: 5 warm-up predicts, then timed predicts (20 in our saved CSV) including hand extraction, on a 640x480 image. It reports mean, median, p90, std, FPS, the pre/infer/post split, params, file size, GFLOPs and the model input shape.

**Q40. Why use median and p90 too, not just the mean?**
The mean can be pulled by a few slow outliers. Median shows the typical case, p90 shows how bad slow frames get, which matters for real time. In our runs they are close (Big@640 on CPU mean 233.6, median 232.8, p90 235.8), so timings were stable.

**Q41. Why did you choose Tiny@320 as the default?**
8.8 ms on CPU, well inside the 33 ms camera frame budget, and 320 px is enough detail at normal sitting distance. 160 is faster but loses the hand more easily when far away. 640 triples the time for little gain in the game.

**Q42. Is a bigger model always worse for the game?**
No. If the hardware is fast enough that the big model still fits in the frame budget, its better accuracy helps (for example when you are far away or the light is bad). The point is to pick the best model that meets the deadline on your hardware.

**Q43. How would you make Big run in real time on an edge device?**
Quantize to INT8, prune channels, distill into a smaller student, export to ONNX/TFLite/CoreML to use optimized runtimes and NPUs, and lower the input size. Or simply choose a smaller model.

**Q44. What is quantization?**
Representing weights and activations with fewer bits, usually INT8 instead of FP32. The model gets about 4x smaller than FP32 and faster on INT8 hardware, with a small accuracy loss. Done after training (post training quantization) or during training (quantization aware training).

**Q45. What is pruning and distillation?**
Pruning removes weights or channels that matter little, then fine tunes. Distillation trains a small student to mimic a large teacher's outputs, so the small model learns more than it would alone.

**Q46. Why ONNX or TFLite?**
They are portable formats with optimized runtimes. ONNX Runtime fuses operations and supports many hardware backends. TFLite runs on phones and small devices with INT8 and GPU/NPU delegates. PyTorch in Python is heavy for an edge board.

**Q47. Why Streamlit?**
It lets us build a UI with sliders, tables and charts in pure Python quickly. The downside is that it reruns the whole script on every interaction, so we cache models with `st.cache_resource`, keep the tracker and history in `st.session_state`, and stop old threads on a new round.

**Q48. Why do you encode JPEG yourself?**
Passing a numpy array to `st.image` encodes at quality 100, which was slow. Encoding at quality 80 with OpenCV is much faster and still looks fine.

**Q49. What does "Hand found %" tell you?**
The share of detector results where a hand was found. Low values mean the model lost you: too far, bad light, elbows out of frame, or too small an input size.

**Q50. What happens if the camera fails?**
`open_camera` retries 30 times. If it never gets a frame we show help about macOS permissions and the camera index. During a round, 30 empty reads in a row end the round with "Lost the camera feed."

**Q51. What is the green sticker mode?**
A fourth tracking option with no neural network. The player sticks a green dot on the palm, and we find it with a colour threshold in HSV, clean the mask with a morphological open, take the biggest contour and use its centroid as the hand position.

**Q52. Why convert to HSV instead of thresholding in RGB?**
In RGB, brightness is spread over all three channels, so the same green in shadow and in bright light are far apart. HSV puts the colour in one channel (hue) and brightness in another (value), so one hue range catches the sticker in most lighting. We still need minimum S and V so grey and very dark pixels do not count.

**Q53. What does `cv2.inRange` give you?**
A binary mask the same size as the image: 255 where all three HSV values are inside the low/high limits, 0 elsewhere.

**Q54. What is a morphological open and why do it?**
Erosion followed by dilation with a small kernel (3x3 here). Erosion deletes blobs smaller than the kernel (noise pixels), dilation grows the surviving blobs back. So single noisy pixels go away and the sticker stays about the same size.

**Q55. How do you get the centre of the blob?**
Image moments of the contour: `m00` is the area, `m10` and `m01` are sums of x and y. Centroid is `(m10 / m00, m01 / m00)`. Then we divide by the scale to go back to full frame pixels.

**Q56. Why is there a minimum area?**
So a few stray green pixels or a tiny far away green thing do not count as the hand. It is a fraction of the image area (0.05 %), so it scales with the input size.

**Q57. How fast is it compared to YOLO?**
In the benchmark green@160 takes 0.06 ms and green@640 0.52 ms, while the fastest YOLO setting (Tiny@160 on CPU) takes 4.3 ms. So about 8x to 70x faster, with 0 params and 0 MB. In the live game both are capped by the camera at about 25 to 30 FPS, but the delay drops from about 12 ms (Tiny@320) to about 1 ms.

**Q58. Why is it so fast?**
No network. Every step is a cheap per pixel operation (resize, blur, colour convert, compare, 3x3 erode/dilate) plus contour tracing, roughly O(pixels), done in optimized C++. At 160 px there are only 19,200 pixels. YOLO Tiny@160 is about 0.5 GFLOPs per frame.

**Q59. Why is green@80 slower than green@160 in your benchmark?**
OpenCV's INTER_AREA resize has a fast path for shrinking by 2x or 4x, and 640 to 80 is 8x, so it goes through the slower general code (about 0.2 ms vs 0.01 ms). It is still far below a millisecond, so it does not matter for the game.

**Q60. If it is so fast, why not always use it?**
It needs a marker and controlled conditions. Any bigger green object (plant, t-shirt, wall) wins, lighting and auto white balance can push the sticker out of the range, and without the sticker it finds nothing. The pose model works for anyone with no setup. It is the speed vs generality trade-off.

**Q61. Classical CV vs deep learning, which is better for edge devices?**
Depends on whether you control the scene. Classical CV is tiny, fast, low power and easy to explain, and fine when the setup is fixed (markers, factory lines). Deep learning handles messy real scenes but costs much more compute and memory. A common edge design uses a cheap classical step first and runs the network only when needed.

**Q62. How would you make the colour tracker more reliable?**
Calibrate the colour at the start (sample the sticker and set the range around it), use a less common marker colour, track the blob over time and ignore jumps (for example only search near the last position), and lock the camera's white balance and exposure if possible.

---

## 7. Limitations

- Our delay number does not include camera capture delay or display delay, so the felt delay is larger.
- The live game is capped by the webcam at about 30 FPS, so it cannot show the full speed difference between fast models.
- 4 CPU threads on a laptop is only a proxy for an edge device. We did not measure on a real Raspberry Pi or Jetson, and we did not measure power or memory.
- The palm point is a guess from the wrist and elbow, not a real hand detector. It fails if the elbows are not visible or the hand is bent strongly at the wrist.
- We did not measure accuracy (for example keypoint error) ourselves. The benchmark only checks whether a hand was found on one easy image (20/20 for all models).
- The benchmark uses a single image, so it does not cover different lighting, distance or many people.
- Game results depend on the player, their reaction time and practice, so rounds are noisy. Several rounds per setting are needed.
- The GIL and Streamlit overheads add some delay in the live loop compared to the benchmark.
- Green sticker mode needs a marker, a fixed hand picked colour range and no other green things in view. We did not test it under different lighting.
- The biggest box rule can pick the wrong person if someone walks close behind the player.

---

## 8. Future work

- Export the models to ONNX and TFLite and compare runtimes.
- INT8 quantization of Tiny and Small, then measure speed, file size and accuracy change.
- Try pruning and distillation (Big as teacher, Tiny as student).
- Run on a real Raspberry Pi or Jetson and measure power and temperature too.
- Measure full glass to glass latency, for example by filming a stopwatch shown on screen next to the game.
- Add a real hand landmark model for a more exact palm position.
- Predict the hand position forward in time (for example with a Kalman filter) to hide some of the delay.
- Measure keypoint accuracy on a labelled set to plot accuracy vs latency for each setting.
- Run detection in a separate process to avoid the GIL completely.

---

## 9. Suggested contribution split

- Ishant Mehndiratta: `detector.py` - loading the models, warm-up and MPS fallback, the hand point logic (person choice, palm trick, hand choice), EMA smoothing, and the `HandTracker` background thread with the latest frame mailbox and frame dropping.
- Satyam Tiwari: `game.py` - fruits, spawning, speed ramp, catch and miss logic, drawing and the HUD, the seeded fair mode, and the test suite in `tests/` (game tests, detector tests with the fake model, app smoke tests).
- Ishaan Sharma: `app.py` - the Streamlit page, sidebar settings, camera handling, the main loop with countdown, the stats panel, the "What the model sees" preview, and the results table, charts and CSV download.
- Hiten Mohindra: `benchmark.py` - the offline timing setup, collecting and checking the numbers in `results/benchmark.csv` and the plot, plus the README and the project report.

Everyone should still be able to explain every part, since the viva can ask anyone about anything.
