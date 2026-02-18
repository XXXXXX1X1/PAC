import argparse
import sys
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

BLOCK_SIZE = 1280  # 80 ms at 16 kHz
SAMPLE_RATE = 16000


def read_threshold(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    threshold = read_threshold(args.threshold)

    model = Model(wakeword_models=[args.model], inference_framework="onnx")

    last_trigger = 0.0
    hit_streak = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
        ) as stream:
            print("Listening... Ctrl+C to stop")
            while True:
                audio, _ = stream.read(BLOCK_SIZE)
                audio = audio.reshape(-1)
                # convert to int16 PCM
                audio_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
                preds = model.predict(audio_i16)
                key = next(iter(preds.keys()))
                score = float(preds[key])
                print(f"score={score:.3f}")

                if score > threshold:
                    hit_streak += 1
                else:
                    hit_streak = 0

                now = time.time()
                if hit_streak >= args.patience and (now - last_trigger) >= 1.0:
                    print("Wakeword Detected!")
                    last_trigger = now
                    hit_streak = 0

    except sd.PortAudioError as e:
        print(
            "Microphone access error. On macOS, grant microphone permission "
            "to your terminal app in System Settings -> Privacy & Security -> Microphone.\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
