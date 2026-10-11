# Latency Limbo

A webcam game where you catch falling fruits with your hand, tracked by a YOLO pose model. Switch to a bigger model and you can actually feel the delay.

Team: Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma, Hiten Mohindra

## What it teaches

- Bigger models are usually more accurate but slower, and in a real time app "slower" means you react to where things were, not where they are.
- Input resolution is a knob too: fewer pixels is faster but the model sees less detail.
- Hardware matters. On small edge devices a tiny fast model often beats a big accurate one.
- Classic CV vs deep learning: the "Green sticker (colour tracking)" mode skips the neural net and just finds a green dot on your palm with an HSV colour threshold. It is way faster (under 1 ms per frame) but only works with the sticker and decent lighting.

## Setup

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python detector.py
```

`python detector.py` downloads the three model weights (about 145 MB total) into `models/` and prints their sizes. You can skip it, the app downloads a model the first time you use it, but then the first round takes a while to start.

## Run

```
streamlit run app.py
```

## How to play

1. Pick a model and input size in the sidebar.
2. Press Start round and sit back so your shoulders and elbows are in the frame.
3. Raise a hand. The white circle is where the model thinks your hand is. Catch the fruits with it.
4. Play a few rounds with different settings and compare them in the Results table.

For the green sticker mode, stick a green dot (or green tape) on your palm and keep green stuff (plants, clothes) out of the picture. It also offers 80 px input.

Keep "Same fruits every round" on if you want a fair comparison between models.

## Benchmark

```
python benchmark.py          # CPU only
python benchmark.py --mps    # also run on the Apple GPU
```

It times every model at every input size, plus the green sticker tracker at 80/160/320/640 px, on a fixed test image and saves `results/benchmark.csv` and `results/benchmark.png`. Run `python benchmark.py --help` for the other options.

## Tests

```
pytest                  # everything
pytest -m "not slow"    # skip the test that loads real weights
```

## Files

- `app.py` - the Streamlit app (UI, camera loop, results table)
- `game.py` - the fruit game logic and drawing, no ML in here
- `detector.py` - loads the pose models (or the green colour tracker) and runs them in a background thread
- `benchmark.py` - offline speed test of all models and input sizes
- `tests/` - pytest tests for the game, the detector helpers and the app page
- `results/` - benchmark output

## Camera troubleshooting

- macOS: the app you launched Streamlit from (Terminal, iTerm, VS Code) needs camera permission in System Settings > Privacy & Security > Camera. Restart that app after allowing it.
- If the permission prompt never showed up, this triggers it once:
  `python -c "import cv2; print(cv2.VideoCapture(0).read()[0])"`
- Wrong camera (for example your iPhone through Continuity Camera)? Change the camera index under Advanced in the sidebar.
- Close Zoom, FaceTime or other browser tabs that might be using the camera.
