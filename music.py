#!/usr/bin/env python3
"""
Neverness to Everness piano helper.
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


NOTE_MIN = 36
NOTE_MAX = 71

SHIFT = "shift_left"
CTRL = "ctrl_left"


TOP_ROW: List[Tuple[str, Optional[str]]] = [
    ("q", None), ("q", SHIFT), ("w", None), ("e", CTRL),
    ("e", None), ("r", None), ("r", SHIFT), ("t", None),
    ("t", SHIFT), ("y", None), ("u", CTRL), ("u", None),
]

MID_ROW: List[Tuple[str, Optional[str]]] = [
    ("a", None), ("a", SHIFT), ("s", None), ("d", CTRL),
    ("d", None), ("f", None), ("f", SHIFT), ("g", None),
    ("g", SHIFT), ("h", None), ("j", CTRL), ("j", None),
]

BOT_ROW: List[Tuple[str, Optional[str]]] = [
    ("z", None), ("z", SHIFT), ("x", None), ("c", CTRL),
    ("c", None), ("v", None), ("v", SHIFT), ("b", None),
    ("b", SHIFT), ("n", None), ("m", CTRL), ("m", None),
]


def create_key_map(mirror_lr: bool = False, swap_top_bottom: bool = False) -> Dict[int, Tuple[str, Optional[str]]]:
    top = TOP_ROW[:]
    mid = MID_ROW[:]
    bot = BOT_ROW[:]

    if swap_top_bottom:
        top, bot = bot, top

    if mirror_lr:
        top.reverse()
        mid.reverse()
        bot.reverse()

    mapping: Dict[int, Tuple[str, Optional[str]]] = {}
    for index, pair in enumerate(bot):
        mapping[36 + index] = pair
    for index, pair in enumerate(mid):
        mapping[48 + index] = pair
    for index, pair in enumerate(top):
        mapping[60 + index] = pair
    return mapping


def densest_center(notes: Sequence[int], width: int = 35) -> float:
    if not notes:
        return (NOTE_MIN + NOTE_MAX) / 2

    counts = collections.Counter(notes)
    unique = sorted(counts)
    fallback = (min(notes) + max(notes)) / 2
    best_center = fallback
    best_count = -1

    for start in unique:
        end = start + width
        current = sum(counts[n] for n in counts if start <= n <= end)
        center = (start + end) / 2
        if current > best_count:
            best_count = current
            best_center = center
        elif current == best_count and abs(center - fallback) < abs(best_center - fallback):
            best_center = center

    return best_center


def fold_to_range(note: int, low: int = NOTE_MIN, high: int = NOTE_MAX) -> int:
    while note < low:
        note += 12
    while note > high:
        note -= 12
    return note


def compress_notes(
    notes: Sequence[int],
    mode: str = "key-preserve",
    transpose: int = 0,
    low: int = NOTE_MIN,
    high: int = NOTE_MAX,
) -> Dict[int, int]:
    unique = sorted(set(notes))
    if not unique:
        return {}

    target_center = (low + high) / 2
    raw_center = densest_center(unique, high - low)

    def octave_only(source: int) -> int:
        for step in range(-6, 7):
            candidate = source + 12 * step
            if low <= candidate <= high:
                return candidate
        return fold_to_range(source, low, high)

    result: Dict[int, int] = {}

    if mode == "direct":
        for note in unique:
            result[note] = fold_to_range(note + transpose, low, high)
        return result

    if mode in {"key-preserve", "key-clip"}:
        shift = int(round(target_center - raw_center)) + transpose
        for note in unique:
            result[note] = fold_to_range(note + shift, low, high)
        return result

    if mode in {"octave-preserve", "octave-clip"}:
        octave_shift = int(round((target_center - raw_center) / 12))
        octave_shift += int(round(transpose / 12))
        for note in unique:
            moved = note + 12 * octave_shift
            if mode == "octave-preserve":
                result[note] = octave_only(moved)
            else:
                result[note] = fold_to_range(moved, low, high)
        return result

    raise ValueError(f"Unknown compression mode: {mode}")


def load_midi_events(
    midi_path: Path,
    mode: str = "key-preserve",
    transpose: int = 0,
    mirror_lr: bool = False,
    swap_top_bottom: bool = False,
    ignore_drum_channel: bool = True,
) -> List[dict]:
    if mido is None:
        raise RuntimeError("mido is not installed. Run: pip install mido")

    midi = mido.MidiFile(str(midi_path), clip=True)
    merged = mido.merge_tracks(midi.tracks)

    tempo = 500000
    elapsed = 0.0
    raw_notes: List[int] = []
    events: List[Tuple[float, str, int, Optional[int]]] = []

    for msg in merged:
        elapsed += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)

        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        if msg.type not in ("note_on", "note_off"):
            continue

        channel = getattr(msg, "channel", None)
        if ignore_drum_channel and channel == 9:
            continue

        is_down = msg.type == "note_on" and msg.velocity > 0
        raw_notes.append(msg.note)
        events.append((elapsed, "down" if is_down else "up", msg.note, channel))

    note_map = compress_notes(raw_notes, mode=mode, transpose=transpose)
    key_map = create_key_map(mirror_lr=mirror_lr, swap_top_bottom=swap_top_bottom)

    output: List[dict] = []
    for timestamp, op, raw_note, _channel in events:
        playable = note_map.get(raw_note)
        if playable is None:
            continue

        pair = key_map.get(playable)
        if pair is None:
            continue

        key, modifier = pair
        if modifier:
            output.append({"time": timestamp, "op": op, "key": modifier})
        output.append({"time": timestamp, "op": op, "key": key})

    output.sort(key=lambda item: (item["time"], 0 if item["op"] == "up" else 1, item["key"]))
    return output


def load_json_events(json_path: Path) -> List[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of events.")

    output: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        timestamp = float(item.get("time", 0.0))
        raw_op = item.get("op")
        key = item.get("key")
        if key is None:
            continue

        op = "down" if raw_op in ("down", "press", "keydown") else "up"
        output.append({"time": timestamp, "op": op, "key": str(key)})

    output.sort(key=lambda item: (item["time"], 0 if item["op"] == "up" else 1, item["key"]))
    return output


class InputBackend:
    def press(self, key: str) -> None:
        raise NotImplementedError

    def release(self, key: str) -> None:
        raise NotImplementedError


class WindowsBackend(InputBackend):
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows backend is only available on Windows.")

        import ctypes

        self.user32 = ctypes.windll.user32
        self.vk_map = {
            SHIFT: 0xA0,
            CTRL: 0xA2,
            "alt_left": 0xA4,
        }
        for ch in "abcdefghijklmnopqrstuvwxyz":
            self.vk_map[ch] = ord(ch.upper())

    def _code(self, key: str) -> int:
        if key not in self.vk_map:
            raise KeyError(f"Unsupported key: {key}")
        return self.vk_map[key]

    def press(self, key: str) -> None:
        self.user32.keybd_event(self._code(key), 0, 0, 0)

    def release(self, key: str) -> None:
        self.user32.keybd_event(self._code(key), 0, 2, 0)


class PynputBackend(InputBackend):
    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key  # type: ignore

        self.keyboard = Controller()
        self.key_enum = Key

    def _translate(self, key: str):
        if key == SHIFT:
            return self.key_enum.shift_l
        if key == CTRL:
            return self.key_enum.ctrl_l
        if key == "alt_left":
            return self.key_enum.alt_l
        return key

    def press(self, key: str) -> None:
        self.keyboard.press(self._translate(key))

    def release(self, key: str) -> None:
        self.keyboard.release(self._translate(key))


def get_backend() -> InputBackend:
    if os.name == "nt":
        try:
            return WindowsBackend()
        except Exception:
            pass

    try:
        return PynputBackend()
    except Exception as exc:
        raise RuntimeError(
            "No usable keyboard backend found. Install pynput, or run on Windows."
        ) from exc


class Player:
    def __init__(
        self,
        events: Sequence[dict],
        backend: InputBackend,
        delay: float = 0.0,
        dry_run: bool = False,
    ) -> None:
        self.events = list(events)
        self.backend = backend
        self.delay = delay
        self.dry_run = dry_run
        self.stop_flag = threading.Event()
        self.held = collections.Counter()
        self.lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None

    def _release_all(self) -> None:
        with self.lock:
            held_snapshot = list(self.held.items())
            self.held.clear()

        for key, count in held_snapshot:
            for _ in range(count):
                try:
                    self.backend.release(key)
                except Exception:
                    pass

    def stop(self) -> None:
        self.stop_flag.set()
        self._release_all()

    def _sleep_until(self, seconds: float) -> bool:
        deadline = time.perf_counter() + max(0.0, seconds)
        while True:
            if self.stop_flag.is_set():
                return False
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.01))

    def _run(self) -> None:
        try:
            if self.delay > 0:
                print(f"Starting in {self.delay:.2f} seconds... (F6 to stop)")
                if not self._sleep_until(self.delay):
                    return

            start = time.perf_counter()
            for event in self.events:
                if self.stop_flag.is_set():
                    break

                target = start + float(event["time"])
                while True:
                    if self.stop_flag.is_set():
                        break
                    remaining = target - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.003))

                if self.stop_flag.is_set():
                    break

                key = event["key"]
                op = event["op"]

                if self.dry_run:
                    print(f"{event['time']:.3f}s {op:>4} {key}")
                    continue

                with self.lock:
                    if op == "down":
                        if self.held[key] == 0:
                            self.backend.press(key)
                        self.held[key] += 1
                    else:
                        if self.held[key] > 0:
                            self.held[key] -= 1
                            if self.held[key] == 0:
                                self.backend.release(key)

        finally:
            self._release_all()
            print("Playback stopped.")

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            print("Already running.")
            return
        self.stop_flag.clear()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        print("Playback thread started.")


def install_hotkeys(player: Player) -> None:
    if kb is None:
        return
    kb.add_hotkey("f5", player.start)
    kb.add_hotkey("f6", player.stop)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play MIDI or JSON piano events into the game.")
    parser.add_argument("input", help="Path to a .mid/.midi or .json event file")
    parser.add_argument(
        "--mode",
        default="key-preserve",
        choices=["key-preserve", "key-clip", "octave-preserve", "octave-clip", "direct"],
        help="How to compress MIDI notes into the playable range",
    )
    parser.add_argument("--transpose", type=int, default=0, help="Shift notes before compression")
    parser.add_argument("--reverse-h", action="store_true", help="Mirror each octave left-to-right")
    parser.add_argument("--reverse-v", action="store_true", help="Swap the top and bottom rows")
    parser.add_argument("--count-in", type=float, default=0.0, help="Delay before playback starts after F5")
    parser.add_argument("--dry-run", action="store_true", help="Print events without sending keys")
    parser.add_argument("--export-json", type=str, default="", help="Export generated events to JSON")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1

    suffix = input_path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        events = load_midi_events(
            input_path,
            mode=args.mode,
            transpose=args.transpose,
            mirror_lr=args.reverse_h,
            swap_top_bottom=args.reverse_v,
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
    player = Player(events, backend, delay=args.count_in, dry_run=args.dry_run)

    print("Ready.")
    print("F5 = start")
    print("F6 = stop")
    print("ESC = quit")

    if kb is None:
        print("The 'keyboard' package is missing. Install it with: pip install keyboard", file=sys.stderr)
        return 1

    install_hotkeys(player)

    try:
        kb.wait("esc")
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
