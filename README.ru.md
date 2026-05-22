# BGE-M3 OpenVINO

Модель эмбеддингов [BGE-M3](https://huggingface.co/BAAI/bge-m3), сконвертированная в формат OpenVINO IR с INT8-квантизацией. Оптимизирована для инференса на Intel NPU/CPU/GPU.

## Быстрый старт

### Установка

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts/requirements.txt
```

### Загрузка и конвертация

```bash
python scripts/export_onnx.py --output models/onnx
python scripts/convert_openvino.py --source models/onnx --output models/bge-m3-int8-ov
```

### Тестирование

```bash
python scripts/test_model.py --model-dir models/bge-m3-int8-ov
```

### Использование

```python
import openvino as ov
import numpy as np
from transformers import AutoTokenizer

core = ov.Core()
model = core.compile_model("models/bge-m3-int8-ov/model.xml", "CPU")
tokenizer = AutoTokenizer.from_pretrained("models/bge-m3-int8-ov")

text = "Что такое BGE M3?"
encoded = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=512)
result = model({
    "input_ids": encoded["input_ids"].astype(np.int64),
    "attention_mask": encoded["attention_mask"].astype(np.int64),
})
embedding = result["sentence_embedding"]  # shape: [1, 1024]
print(embedding.shape)
```

## Параметры модели

| Свойство | Значение |
|----------|----------|
| Базовая модель | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Лицензия | MIT |
| Входная форма | [1, 512] (статическая) |
| Входы | `input_ids` (int64), `attention_mask` (int64) |
| Выход | `sentence_embedding` [1, 1024] float32 |
| Квантизация | INT8 asymmetric (NNCF) |
| Размерность | 1024 |
| Макс. длина | 512 токенов |

## Пайплайн конвертации

Проект содержит три скрипта, выполняющих полный цикл:

| Шаг | Скрипт | Вход | Выход |
|-----|--------|------|-------|
| 1. ONNX экспорт | `export_onnx.py` | `BAAI/bge-m3` с Hugging Face | `model.onnx` [1,512] |
| 2. OpenVINO + INT8 | `convert_openvino.py` | `model.onnx` | `model.xml`/`model.bin` INT8 |
| 3. Тестирование | `test_model.py` | OpenVINO IR | Проверка формы, dtype, L2-нормы |

## Структура проекта

```
bge-m3-openvino/
├── scripts/
│   ├── export_onnx.py            # PyTorch → ONNX
│   ├── convert_openvino.py       # ONNX → OpenVINO IR + INT8 NNCF
│   ├── test_model.py             # Валидация модели
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD
├── docs/
│   └── model_preparation.md      # Подробная документация
├── .gitignore
├── LICENSE                       # MIT
├── README.md                     # English
└── README.ru.md                  # Русский
```

## CI/CD

При пуше в ветку `main` GitHub Actions автоматически:

1. Загружает оригинальную модель `BAAI/bge-m3` с Hugging Face
2. Экспортирует в ONNX
3. Конвертирует в OpenVINO IR с INT8-квантизацией
4. Запускает тесты
5. Публикует бандл модели на Hugging Face

### Необходимые секреты

Настройте в репозитории GitHub **Settings → Secrets and variables → Actions**:

| Секрет | Описание |
|--------|----------|
| `HF_TOKEN` | Токен Hugging Face с правами записи |
| `HF_REPO` | Целевой репозиторий (например, `your-username/bge-m3-int8-ov`) |

### Ручной запуск

**Actions → Convert and Publish to Hugging Face → Run workflow**.

## Состав бандла

| Файл | Описание |
|------|----------|
| `model.xml` | OpenVINO IR граф |
| `model.bin` | OpenVINO IR веса (INT8, ~543 MB) |
| `tokenizer.json` | Токенизатор Hugging Face |
| `tokenizer_config.json` | Конфигурация токенизатора |
| `sentencepiece.bpe.model` | SentencePiece модель |
| `special_tokens_map.json` | Специальные токены |
| `config.json` | Конфигурация XLM-RoBERTa |

## Сигнатура модели

| Направление | Имя | Форма | Тип |
|-------------|-----|-------|-----|
| Вход | `input_ids` | [1, 512] | int64 |
| Вход | `attention_mask` | [1, 512] | int64 |
| Выход | `token_embeddings` | [1, 512, 1024] | float32 |
| Выход | `sentence_embedding` | [1, 1024] | float32 |

## Лицензия

MIT. Оригинальная модель — [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), Beijing Academy of Artificial Intelligence.

## Ссылки

- [Оригинальная модель](https://huggingface.co/BAAI/bge-m3)
- [Статья BGE-M3](https://arxiv.org/abs/2402.03216)
- [ai2npu — NPU инференс на C++](https://github.com/...)
