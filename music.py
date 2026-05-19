# Neverness to Everness Piano Player
# F5 = Start
# F6 = Stop
# ESC = Exit

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import mido  # type: ignore
except ImportError:  # pragma: no cover
    mido = None

try:
    import keyboard as kb  # type: ignore
except Exception:  # pragma: no cover
    kb = None


NOTE_LOW = 36
NOTE_HIGH = 71

SHIFT_KEY = "shift_left"
CTRL_KEY = "ctrl_left"


# ------------------------------ Layout tables ------------------------------

LOW_ROW: List[Tuple[str, Optional[str]]] = [
    ("z", None), ("z", SHIFT_KEY), ("x", None), ("c", CTRL_KEY),
    ("c", None), ("v", None), ("v", SHIFT_KEY), ("b", None),
    ("b", SHIFT_KEY), ("n", None), ("m", CTRL_KEY), ("m", None),
]

MID_ROW: List[Tuple[str, Optional[str]]] = [
    ("a", None), ("a", SHIFT_KEY), ("s", None), ("d", CTRL_KEY),
    ("d", None), ("f", None), ("f", SHIFT_KEY), ("g", None),
    ("g", SHIFT_KEY), ("h", None), ("j", CTRL_KEY), ("j", None),
]

HIGH_ROW: List[Tuple[str, Optional[str]]] = [
    ("q", None), ("q", SHIFT_KEY), ("w", None), ("e", CTRL_KEY),
    ("e", None), ("r", None), ("r", SHIFT_KEY), ("t", None),
    ("t", SHIFT_KEY), ("y", None), ("u", CTRL_KEY), ("u", None),
]


def build_note_mapping(reverse_h: bool = False, reverse_v: bool = False) -> Dict[int, Tuple[str, Optional[str]]]:
    """Create the game's 36..71 note-to-key map."""
    low = LOW_ROW[:]
    mid = MID_ROW[:]
    high = HIGH_ROW[:]

    if reverse_v:
        low, high = high, low

    if reverse_h:
        low = list(reversed(low))
        mid = list(reversed(mid))
        high = list(reversed(high))

    mapping: Dict[int, Tuple[str, Optional[str]]] = {}
    for offset, pair in enumerate(low):
        mapping[36 + offset] = pair
    for offset, pair in enumerate(mid):
        mapping[48 + offset] = pair
    for offset, pair in enumerate(high):
        mapping[60 + offset] = pair
    return mapping


# ------------------------------ Compression helpers ------------------------------

def dense_center(notes: Sequence[int], window: int = 35) -> float:
    """Find the center of the densest window."""
    if not notes:
        return (NOTE_LOW + NOTE_HIGH) / 2

    counts = collections.Counter(notes)
    values = sorted(counts)
    fallback = (min(notes) + max(notes)) / 2
    best = fallback
    best_count = -1

    for start in values:
        end = start + window
        current = sum(counts[n] for n in counts if start <= n <= end)
        center = (start + end) / 2
        if current > best_count:
            best_count = current
            best = center
        elif current == best_count and abs(center - fallback) < abs(best - fallback):
            best = center

    return best


def fold_into_bounds(note: int, low: int = NOTE_LOW, high: int = NOTE_HIGH) -> int:
    while note < low:
        note += 12
    while note > high:
        note -= 12
    return note


def compress_notes(
    notes: Sequence[int],
    mode: str = "key-preserve",
    transpose: int = 0,
    low: int = NOTE_LOW,
    high: int = NOTE_HIGH,
) -> Dict[int, int]:
    """Return {raw_note: playable_note}."""
    unique = sorted(set(notes))
    if not unique:
        return {}

    target_center = (low + high) / 2
    raw_center = dense_center(unique, high - low)

    def octave_shift_only(source: int) -> int:
        for step in range(-6, 7):
            placed = source + 12 * step
            if low <= placed <= high:
                return placed
        return fold_into_bounds(source, low, high)

    result: Dict[int, int] = {}

    if mode == "direct":
        for n in unique:
            result[n] = fold_into_bounds(n + transpose, low, high)
        return result

    if mode in {"key-preserve", "key-clip"}:
        shift = int(round(target_center - raw_center)) + transpose
        for n in unique:
            result[n] = fold_into_bounds(n + shift, low, high)
        return result

    if mode in {"octave-preserve", "octave-clip"}:
        shift = int(round((target_center - raw_center) / 12))
        shift += int(round(transpose / 12))
        for n in unique:
            moved = n + 12 * shift
            result[n] = octave_shift_only(moved) if mode == "octave-preserve" else fold_into_bounds(moved, low, high)
        return result

    raise ValueError(f"Unknown compression mode: {mode}")


# ------------------------------ Event loading ------------------------------

def load_midi_events(
    midi_path: Path,
    mode: str = "key-preserve",
    transpose: int = 0,
    reverse_h: bool = False,
    reverse_v: bool = False,
    ignore_channel_9: bool = True,
) -> List[dict]:
    """Convert a MIDI file into sorted key events."""
    if mido is None:
        raise RuntimeError("mido is not installed. Run: pip install mido")

    mid = mido.MidiFile(str(midi_path), clip=True)
    merged = mido.merge_tracks(mid.tracks)

    tempo = 500000
    elapsed = 0.0
    raw_notes: List[int] = []
    merged_events: List[Tuple[float, str, int, Optional[int]]] = []

    for msg in merged:
        elapsed += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type not in ("note_on", "note_off"):
            continue

        channel = getattr(msg, "channel", None)
        if ignore_channel_9 and channel == 9:
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            raw_notes.append(msg.note)
            merged_events.append((elapsed, "down", msg.note, channel))
        else:
            raw_notes.append(msg.note)
            merged_events.append((elapsed, "up", msg.note, channel))

    note_map = compress_notes(raw_notes, mode=mode, transpose=transpose)
    note_to_key = build_note_mapping(reverse_h=reverse_h, reverse_v=reverse_v)

    out: List[dict] = []
    for t, op, raw_note, _channel in merged_events:
        playable = note_map.get(raw_note)
        if playable is None:
            continue
        pair = note_to_key.get(playable)
        if pair is None:
            continue
        key, modifier = pair
        if modifier:
            out.append({"time": t, "op": op, "key": modifier})
        out.append({"time": t, "op": op, "key": key})

    out.sort(key=lambda e: (e["time"], 0 if e["op"] == "up" else 1, e["key"]))
    return out


def load_json_events(json_path: Path) -> List[dict]:
    """Load prebuilt event format from JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of events.")

    out: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        t = float(item.get("时间", item.get("time", 0.0)))
        op_raw = item.get("操作", item.get("op"))
        key = item.get("按键", item.get("key"))
        if key is None:
            continue
        op = "down" if op_raw in ("按下", "down", "press", "keydown") else "up"
        out.append({"time": t, "op": op, "key": str(key)})

    out.sort(key=lambda e: (e["time"], 0 if e["op"] == "up" else 1, e["key"]))
    return out


# ------------------------------ Input backends ------------------------------

class InputBackend:
    def press(self, key: str) -> None:
        raise NotImplementedError

    def release(self, key: str) -> None:
        raise NotImplementedError


class BundleBackend(InputBackend):
    def __init__(self) -> None:
        from importlib import import_module

        module = import_module("异环游戏特殊键盘")
        self._press = getattr(module, "模拟按键按下")
        self._release = getattr(module, "模拟按键弹起")

    def press(self, key: str) -> None:
        self._press(key)

    def release(self, key: str) -> None:
        self._release(key)


class WindowsCtypesBackend(InputBackend):
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows ctypes backend is only available on Windows.")

        import ctypes

        self.user32 = ctypes.windll.user32
        self.vk = {
            SHIFT_KEY: 0xA0,
            CTRL_KEY: 0xA2,
            "alt_left": 0xA4,
        }
        for letter in "abcdefghijklmnopqrstuvwxyz":
            self.vk[letter] = ord(letter.upper())

    def _vk_code(self, key: str) -> int:
        if key not in self.vk:
            raise KeyError(f"Unsupported key: {key}")
        return self.vk[key]

    def press(self, key: str) -> None:
        self.user32.keybd_event(self._vk_code(key), 0, 0, 0)

    def release(self, key: str) -> None:
        self.user32.keybd_event(self._vk_code(key), 0, 2, 0)


class PynputBackend(InputBackend):
    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key  # type: ignore

        self.keyboard = Controller()
        self.Key = Key

    def _map(self, key: str):
        if key == SHIFT_KEY:
            return self.Key.shift_l
        if key == CTRL_KEY:
            return self.Key.ctrl_l
        if key == "alt_left":
            return self.Key.alt_l
        return key

    def press(self, key: str) -> None:
        self.keyboard.press(self._map(key))

    def release(self, key: str) -> None:
        self.keyboard.release(self._map(key))


def get_backend() -> InputBackend:
    try:
        return BundleBackend()
    except Exception:
        pass

    if os.name == "nt":
        try:
            return WindowsCtypesBackend()
        except Exception:
            pass

    try:
        return PynputBackend()
    except Exception as exc:
        raise RuntimeError(
            "No usable keyboard backend found. Install the bundle module, or pynput, or run on Windows."
        ) from exc


# ------------------------------ Playback ------------------------------

class PlaybackController:
    def __init__(
        self,
        events: Sequence[dict],
        backend: InputBackend,
        count_in: float = 0.0,
        dry_run: bool = False,
    ) -> None:
        self.events = list(events)
        self.backend = backend
        self.count_in = count_in
        self.dry_run = dry_run
        self.stop_event = threading.Event()
        self.active_counts = collections.Counter()
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def release_all(self) -> None:
        with self.lock:
            active_items = list(self.active_counts.items())
            self.active_counts.clear()

        for key, count in active_items:
            for _ in range(count):
                try:
                    self.backend.release(key)
                except Exception:
                    pass

    def stop(self) -> None:
        self.stop_event.set()
        self.release_all()

    def _wait_interruptible(self, seconds: float) -> bool:
        deadline = time.perf_counter() + max(0.0, seconds)
        while True:
            if self.stop_event.is_set():
                return False
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.01))

    def run(self) -> None:
        try:
            if self.count_in > 0:
                print(f"Starting in {self.count_in:.2f} seconds... (F6 to stop)")
                if not self._wait_interruptible(self.count_in):
                    return

            start = time.perf_counter()

            for ev in self.events:
                if self.stop_event.is_set():
                    break

                target_time = start + float(ev["time"])
                while True:
                    if self.stop_event.is_set():
                        break
                    remaining = target_time - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.003))

                if self.stop_event.is_set():
                    break

                key = ev["key"]
                op = ev["op"]

                if self.dry_run:
                    print(f"{ev['time']:.3f}s {op:>4} {key}")
                    continue

                with self.lock:
                    if op == "down":
                        if self.active_counts[key] == 0:
                            self.backend.press(key)
                        self.active_counts[key] += 1
                    elif self.active_counts[key] > 0:
                        self.active_counts[key] -= 1
                        if self.active_counts[key] == 0:
                            self.backend.release(key)

        finally:
            self.release_all()
            print("Playback stopped.")

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            print("Already running.")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        print("Playback thread started.")


# ------------------------------ Hotkeys ------------------------------

def setup_hotkeys(controller: PlaybackController) -> None:
    if kb is None:
        return

    # Use lowercase key names for better compatibility across platforms.
    kb.on_press_key("f5", lambda _event: controller.start())
    kb.on_press_key("f6", lambda _event: controller.stop())


# ------------------------------ CLI ------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play MIDI or JSON piano events into the game.")
    parser.add_argument("input", help="Path to .mid or .json event file")
    parser.add_argument(
        "--mode",
        default="key-preserve",
        choices=["key-preserve", "key-clip", "octave-preserve", "octave-clip", "direct"],
        help="How to compress MIDI notes into the 36..71 game range",
    )
    parser.add_argument("--transpose", type=int, default=0, help="Shift all MIDI notes before compression (semitones)")
    parser.add_argument("--reverse-h", action="store_true", help="Mirror each octave left/right")
    parser.add_argument("--reverse-v", action="store_true", help="Swap low/high octaves")
    parser.add_argument("--count-in", type=float, default=0.0, help="Delay before playback starts after F5")
    parser.add_argument("--dry-run", action="store_true", help="Print events without sending keys")
    parser.add_argument("--export-json", type=str, default="", help="Export generated events to a JSON file")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1

    suffix = input_path.suffix.lower()
    if suffix in (".mid", ".midi"):
        events = load_midi_events(
            input_path,
            mode=args.mode,
            transpose=args.transpose,
            reverse_h=args.reverse_h,
            reverse_v=args.reverse_v,
        )
    elif suffix == ".json":
        events = load_json_events(input_path)
    else:
        print("Input must be a .mid/.midi or .json file", file=sys.stderr)
        return 1

    if not events:
        print("No playable events found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(events)} events from {input_path.name}")

    if args.export_json:
        export_path = Path(args.export_json)
        export_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Exported event file to: {export_path}")

    backend = get_backend()
    controller = PlaybackController(events, backend, count_in=args.count_in, dry_run=args.dry_run)

    print("Ready.")
    print("F5 = start")
    print("F6 = stop")
    print("ESC = quit")

    if kb is None:
        print("The 'keyboard' package is missing. Install it with: pip install keyboard", file=sys.stderr)
        return 1

    setup_hotkeys(controller)

    try:
        kb.wait("esc")
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
