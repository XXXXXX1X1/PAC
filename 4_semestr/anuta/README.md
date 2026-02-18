# anyuta-openwakeword

Минимальный офлайн‑проект: генерация датасета → извлечение фич openWakeWord → обучение → экспорт ONNX → детекция с микрофона.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Генерация датасета

```bash
python3 generate_dataset.py --out data --count-pos 400 --count-neg 800 --seconds 2.0 --sr 16000
```

Файлы появятся в `data/pos/*.wav` и `data/neg/*.wav`.

## Обучение

```bash
python3 train_anyuta.py --data data --out models
```

Результаты:
- `models/anyuta.onnx`
- `models/anyuta_threshold.txt`
- `models/label.txt`

## Запуск детекции (микрофон)

```bash
python3 run_mic.py --model models/anyuta.onnx --threshold models/anyuta_threshold.txt
```

## Типичные проблемы

- **Нет микрофона / доступ запрещен**: на macOS дайте терминалу доступ в System Settings → Privacy & Security → Microphone.
- **Неверный sample rate**: используйте 16kHz, mono, int16 PCM.
- **Неправильный формат WAV**: убедитесь, что файлы `mono/16k/int16`.
- **Задержки/рывки аудио**: попробуйте уменьшить нагрузку или закрыть лишние приложения.
