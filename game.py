"""
Game logic for Latency Limbo.

This file knows nothing about neural networks. It just gets a hand position
(or None) every frame and moves fruits around. Keeping it separate means we
can test the game without a webcam.
"""

import random
from dataclasses import dataclass

import cv2

# colours are BGR because OpenCV
FRUIT_TYPES = {
    "apple":  {"color": (40, 40, 220),  "radius": 22},
    "orange": {"color": (0, 140, 255),  "radius": 22},
    "lime":   {"color": (60, 200, 80),  "radius": 18},
    "grape":  {"color": (150, 50, 130), "radius": 15},
}

BASKET_RADIUS = 50  # how close (in pixels) the hand has to be to catch a fruit

# difficulty knobs, we tweaked these by playing a lot
SPAWN_START = 1.1    # seconds between fruits at the start
SPAWN_DROP = 0.5     # how much that gap shrinks by the end of the round
BASE_SPEED = 170     # pixels per second
SPEED_RAMP = 90      # extra speed added by the end of the round
SPEED_SPREAD = 60    # random extra so fruits don't all fall the same


@dataclass
class Fruit:
    x: float
    y: float
    speed: float  # pixels per second
    kind: str

    @property
    def radius(self):
        return FRUIT_TYPES[self.kind]["radius"]


class FruitGame:
    def __init__(self, width, height, duration=45, seed=None):
        self.width = width
        self.height = height
        self.duration = duration
        self.rng = random.Random(seed)

        self.fruits = []
        self.caught = 0
        self.missed = 0
        self.elapsed = 0.0
        self.time_to_next_spawn = 0.5
        # little "+1" / "miss" popups so the player sees what happened
        self.popups = []

    @property
    def over(self):
        return self.elapsed >= self.duration

    @property
    def time_left(self):
        return max(0.0, self.duration - self.elapsed)

    def spawn_interval(self):
        # starts at one fruit every ~1.1 s and gets a bit faster over the round
        progress = self.elapsed / self.duration
        return SPAWN_START - SPAWN_DROP * progress

    def spawn_fruit(self):
        kind = self.rng.choice(list(FRUIT_TYPES))
        margin = 40
        x = self.rng.uniform(margin, self.width - margin)
        # speed also ramps up a little with time
        base = BASE_SPEED + SPEED_RAMP * (self.elapsed / self.duration)
        speed = self.rng.uniform(base, base + SPEED_SPREAD)
        self.fruits.append(Fruit(x=x, y=-20, speed=speed, kind=kind))

    def update(self, dt, hand):
        """Move the game forward by dt seconds.

        hand is (x, y) in pixels or None if the detector can't see a hand.
        """
        if self.over:
            return

        self.elapsed += dt

        self.time_to_next_spawn -= dt
        if self.time_to_next_spawn <= 0:
            self.spawn_fruit()
            self.time_to_next_spawn = self.spawn_interval()

        still_falling = []
        for fruit in self.fruits:
            fruit.y += fruit.speed * dt

            if hand is not None:
                dx = fruit.x - hand[0]
                dy = fruit.y - hand[1]
                if dx * dx + dy * dy <= (BASKET_RADIUS + fruit.radius) ** 2:
                    self.caught += 1
                    self.popups.append([fruit.x, fruit.y, "+1", 0.6])
                    continue

            if fruit.y - fruit.radius > self.height:
                self.missed += 1
                self.popups.append([fruit.x, self.height - 20, "miss", 0.6])
                continue

            still_falling.append(fruit)
        self.fruits = still_falling

        for p in self.popups:
            p[3] -= dt
        self.popups = [p for p in self.popups if p[3] > 0]

    def catch_rate(self):
        total = self.caught + self.missed
        if total == 0:
            return 0.0
        return self.caught / total

    # ---------------- drawing ----------------

    def draw(self, frame, hand, hand_age_ms=None):
        """Draw fruits, the basket and the HUD on top of the camera frame."""
        for fruit in self.fruits:
            center = (int(fruit.x), int(fruit.y))
            color = FRUIT_TYPES[fruit.kind]["color"]
            cv2.circle(frame, center, fruit.radius, color, -1)
            cv2.circle(frame, center, fruit.radius, (30, 30, 30), 2)
            # tiny stem so they look a bit like fruit
            cv2.line(frame, (center[0], center[1] - fruit.radius),
                     (center[0] + 4, center[1] - fruit.radius - 8), (20, 70, 20), 3)

        if hand is not None:
            hx, hy = int(hand[0]), int(hand[1])
            cv2.circle(frame, (hx, hy), BASKET_RADIUS, (255, 255, 255), 3)
            cv2.circle(frame, (hx, hy), 5, (255, 255, 255), -1)
            if hand_age_ms is not None:
                cv2.putText(frame, f"{hand_age_ms:.0f} ms old", (hx - 45, hy + BASKET_RADIUS + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "no hand found - raise your hand!", (20, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for x, y, text, life in self.popups:
            color = (0, 255, 0) if text == "+1" else (0, 0, 255)
            cv2.putText(frame, text, (int(x) - 15, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        hud = f"Caught: {self.caught}   Missed: {self.missed}   Time: {self.time_left:4.1f}s"
        cv2.rectangle(frame, (0, 0), (self.width, 32), (0, 0, 0), -1)
        cv2.putText(frame, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame
