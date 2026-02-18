import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def safe_normalize_int16(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.int16)
    peak = np.max(np.abs(x))
    if peak == 0:
        return x.astype(np.int16)
    x = x / peak
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def record_samples(out_dir: str, count: int, seconds: float, sr: int, prompt: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    print(prompt)
    print(f"Output: {out_dir}")

    try:
        for i in range(1, count + 1):
            print(f"[{i}/{count}] ready...")
            time.sleep(0.3)

            audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
            sd.wait()
            audio = audio.reshape(-1)

            audio = safe_normalize_int16(audio.astype(np.float32))
            filename = os.path.join(out_dir, f"{i:06d}.wav")
            wavfile.write(filename, sr, audio)
            print(f"saved {filename}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 1
    except sd.PortAudioError as e:
        print(
            "Microphone access error. On macOS, grant microphone permission "
            "to your terminal app in System Settings -> Privacy & Security -> Microphone.\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        return 2

    return 0


def folder_stats(path: str) -> tuple[int, float, float, float]:
    if not os.path.isdir(path):
        return 0, 0.0, 0.0, 0.0

    files = sorted(f for f in os.listdir(path) if f.lower().endswith(".wav"))
    count = 0
    total_sec = 0.0
    rms_values = []

    for name in files:
        fp = os.path.join(path, name)
        try:
            sr, data = wavfile.read(fp)
        except Exception:
            continue
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        count += 1
        total_sec += len(data) / float(sr)
        if data.size:
            rms = np.sqrt(np.mean(data ** 2))
            rms_values.append(rms)

    if rms_values:
        rms_mean = float(np.mean(rms_values))
        rms_std = float(np.std(rms_values))
    else:
        rms_mean = 0.0
        rms_std = 0.0

    return count, total_sec, rms_mean, rms_std


def cmd_stats(pos_dir: str, neg_dir: str) -> int:
    pos = folder_stats(pos_dir)
    neg = folder_stats(neg_dir)

    print(f"pos: {pos_dir}")
    print(f"count={pos[0]} total_sec={pos[1]:.2f} rms_mean={pos[2]:.2f} rms_std={pos[3]:.2f}")
    print(f"neg: {neg_dir}")
    print(f"count={neg[0]} total_sec={neg[1]:.2f} rms_mean={neg[2]:.2f} rms_std={neg[3]:.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    rec = subparsers.add_parser("record")
    rec.add_argument("--out", required=True)
    rec.add_argument("--count", type=int, required=True)
    rec.add_argument("--seconds", type=float, required=True)
    rec.add_argument("--sr", type=int, default=16000)
    rec.add_argument("--prompt", required=True)

    st = subparsers.add_parser("stats")
    st.add_argument("--pos", required=True)
    st.add_argument("--neg", required=True)

    args = parser.parse_args()

    if args.command == "record":
        return record_samples(args.out, args.count, args.seconds, args.sr, args.prompt)
    if args.command == "stats":
        return cmd_stats(args.pos, args.neg)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
