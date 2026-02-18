import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_SR = 16000
CLIP_SECONDS = 2.0

POS_TEXT = "Анюта"
NEG_TEXTS = [
    "Анна",
    "Аня",
    "Антон",
    "А ну",
    "Ну да",
    "Анютка",
    "Сегодня хорошая погода",
    "Мне нужно немного времени",
    "Мы идем домой",
    "Пожалуйста, повторите еще раз",
]

PIPER_VOICES = [
    "ru_RU-irina-medium",
    "ru_RU-dmitri-medium",
    "ru_RU-denis-medium",
    "ru_RU-ruslan-medium",
]


def run_piper(text: str, voice: str, out_wav: Path) -> None:
    # piper-tts CLI
    cmd = [
        "piper",
        "--voice",
        voice,
        "--output_file",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    return sr, data


def to_mono_int16(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    return data


def resample_to_16k(sr: int, data: np.ndarray) -> np.ndarray:
    if sr == TARGET_SR:
        return data
    data_f = data.astype(np.float32)
    g = np.gcd(sr, TARGET_SR)
    up = TARGET_SR // g
    down = sr // g
    resampled = resample_poly(data_f, up, down)
    return resampled.astype(np.int16)


def pad_or_trim(data: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    target_len = int(sr * seconds)
    if len(data) > target_len:
        return data[:target_len]
    if len(data) < target_len:
        pad = target_len - len(data)
        pre = random.randint(0, pad)
        post = pad - pre
        return np.pad(data, (pre, post), mode="constant")
    return data


def time_stretch(data: np.ndarray, stretch: float) -> np.ndarray:
    # naive resample-based stretch
    n = int(len(data) / stretch)
    if n <= 1:
        return data
    idx = np.linspace(0, len(data) - 1, n)
    stretched = np.interp(idx, np.arange(len(data)), data).astype(np.int16)
    return stretched


def add_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(len(data)).astype(np.float32)
    noise = noise / (np.max(np.abs(noise)) + 1e-9)
    noisy = data.astype(np.float32) + noise * (noise_level * 32767.0)
    return np.clip(noisy, -32768, 32767).astype(np.int16)


def normalize_int16(data: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(data)) if len(data) else 0
    if peak == 0:
        return data.astype(np.int16)
    x = data.astype(np.float32) / peak
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def synth_music_like_noise(sr: int, seconds: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    freqs = [220, 330, 440, 550]
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return (sig * 2000).astype(np.int16)


def make_sample(text: str, voice: str, out_path: Path, seconds: float) -> None:
    tmp = out_path.with_suffix(".tmp.wav")
    run_piper(text, voice, tmp)
    sr, data = read_wav(tmp)
    tmp.unlink(missing_ok=True)

    data = to_mono_int16(data)
    data = resample_to_16k(sr, data)

    # augmentations
    if random.random() < 0.7:
        stretch = random.uniform(0.95, 1.05)
        data = time_stretch(data, stretch)
    if random.random() < 0.6:
        gain = random.uniform(0.7, 1.1)
        data = np.clip(data.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    if random.random() < 0.4:
        data = add_noise(data, noise_level=random.uniform(0.01, 0.04))

    data = pad_or_trim(data, TARGET_SR, seconds)
    data = normalize_int16(data)
    wavfile.write(out_path, TARGET_SR, data)


def make_noise_only(out_path: Path, seconds: float) -> None:
    data = np.zeros(int(TARGET_SR * seconds), dtype=np.int16)
    mode = random.choice(["white", "pink", "music", "silence"])
    if mode == "white":
        data = add_noise(data, noise_level=random.uniform(0.02, 0.06))
    elif mode == "pink":
        # simple 1/f noise via cumulative sum
        noise = np.random.randn(len(data)).astype(np.float32)
        noise = np.cumsum(noise)
        noise = noise / (np.max(np.abs(noise)) + 1e-9)
        data = (noise * 2000).astype(np.int16)
    elif mode == "music":
        data = synth_music_like_noise(TARGET_SR, seconds)
    else:
        data = data
    wavfile.write(out_path, TARGET_SR, data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--count-pos", type=int, default=400)
    parser.add_argument("--count-neg", type=int, default=800)
    parser.add_argument("--seconds", type=float, default=CLIP_SECONDS)
    parser.add_argument("--sr", type=int, default=TARGET_SR)
    args = parser.parse_args()

    if args.sr != TARGET_SR:
        print("This script always writes 16kHz output.")

    out_dir = Path(args.out)
    pos_dir = out_dir / "pos"
    neg_dir = out_dir / "neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # Positives
    for i in tqdm(range(1, args.count_pos + 1), desc="pos"):
        voice = random.choice(PIPER_VOICES)
        out_path = pos_dir / f"{i:06d}.wav"
        make_sample(POS_TEXT, voice, out_path, args.seconds)

    # Negatives: mix TTS and synthetic noise
    for i in tqdm(range(1, args.count_neg + 1), desc="neg"):
        out_path = neg_dir / f"{i:06d}.wav"
        if random.random() < 0.7:
            text = random.choice(NEG_TEXTS)
            voice = random.choice(PIPER_VOICES)
            make_sample(text, voice, out_path, args.seconds)
        else:
            make_noise_only(out_path, args.seconds)

    print(f"Done. pos={args.count_pos} neg={args.count_neg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
