import numpy as np

from game import BASKET_RADIUS, Fruit, FruitGame


def make_game(**kw):
    g = FruitGame(640, 480, **kw)
    g.time_to_next_spawn = 999  # we place fruits by hand in these tests
    return g


def test_hand_on_fruit_catches_it():
    g = make_game()
    g.fruits.append(Fruit(x=300, y=200, speed=0, kind="apple"))
    g.update(0.016, (300, 200))
    assert g.caught == 1
    assert g.fruits == []


def test_no_hand_never_catches():
    g = make_game()
    g.fruits.append(Fruit(x=300, y=200, speed=0, kind="apple"))
    g.update(0.016, None)
    assert g.caught == 0
    assert len(g.fruits) == 1


def test_far_hand_does_not_catch():
    g = make_game()
    g.fruits.append(Fruit(x=300, y=200, speed=0, kind="apple"))
    g.update(0.016, (300 + BASKET_RADIUS + 100, 200))
    assert g.caught == 0


def test_fruit_below_bottom_is_a_miss():
    g = make_game()
    g.fruits.append(Fruit(x=300, y=480 + 50, speed=0, kind="apple"))
    g.update(0.016, None)
    assert g.missed == 1
    assert g.fruits == []


def test_timer_ends_game_and_freezes_it():
    g = make_game(duration=1)
    for _ in range(20):
        g.update(0.1, None)
    assert g.over
    assert g.time_left == 0
    g.fruits.append(Fruit(x=300, y=200, speed=0, kind="apple"))
    g.update(0.1, (300, 200))
    assert g.caught == 0 and g.missed == 0
    assert len(g.fruits) == 1


def test_catch_rate():
    g = make_game()
    assert g.catch_rate() == 0.0
    g.caught, g.missed = 3, 3
    assert g.catch_rate() == 0.5


def test_first_fruit_after_half_a_second():
    g = FruitGame(640, 480, seed=1)
    g.update(0.4, None)
    assert len(g.fruits) == 0
    g.update(0.11, None)
    assert len(g.fruits) == 1


def fruit_list(seed):
    g = FruitGame(640, 480, seed=seed)
    seen = []
    for _ in range(100):
        g.update(0.05, None)
        seen.extend((round(f.x, 3), round(f.speed, 3), f.kind) for f in g.fruits)
    return seen


def test_same_seed_same_fruits():
    assert fruit_list(42) == fruit_list(42)


def test_different_seed_different_fruits():
    assert fruit_list(1) != fruit_list(2)


def test_draw_keeps_shape():
    g = make_game()
    g.fruits.append(Fruit(x=300, y=200, speed=0, kind="grape"))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = g.draw(frame, (100, 100), hand_age_ms=80)
    assert out is frame
    assert out.shape == (480, 640, 3)
    out = g.draw(frame, None)
    assert out.shape == (480, 640, 3)
    assert frame.any()  # something actually got drawn
