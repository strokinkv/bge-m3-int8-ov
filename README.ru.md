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
result = model({"input_ids": encoded["input_ids"].astype(np.int64), "attention_mask": encoded["attention_mask"].astype(np.int64)})
embedding = result["sentence_embedding"]  # shape: [1, 1024]
```

## Совместимость с NPU

OpenVINO IR поддерживает динамические формы входов, но компиляция для Intel NPU сейчас требует статических форм. Прямой экспорт через `optimum-cli export openvino` создавал входы `[-1, -1]` без верхних границ: модель работала на CPU, но компиляция на NPU завершалась ошибкой `ov_core_compile_model failed with status -1`.

Поэтому проект экспортирует BGE-M3 через статический ONNX-граф и конвертирует его в INT8 OpenVINO IR с фиксированными входами `[1, 512]`. Граф сразу возвращает CLS-токен как `sentence_embedding [1, 1024]`, поэтому ai2npu не нужны `reshape` и постобработка `last_hidden_state`.

## Параметры модели

| Свойство | Значение |
|----------|----------|
| Целевая модель | [strokinkv/bge-m3-int8-ov](https://huggingface.co/strokinkv/bge-m3-int8-ov) |
| Базовая модель | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Лицензия | MIT |
| Входная форма | [1, 512] (статическая) |
| Входы | `input_ids` (int64), `attention_mask` (int64) |
| Выход | `sentence_embedding` [1, 1024] float32 |
| Квантизация | INT8 asymmetric (NNCF) |
| Размерность | 1024 |
| Макс. длина | 512 токенов |

## Структура проекта

```
bge-m3-openvino/
├── scripts/
│   ├── export_onnx.py            # PyTorch → статический ONNX
│   ├── convert_openvino.py       # ONNX → OpenVINO IR + INT8 NNCF
│   ├── test_model.py             # Валидация модели
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD
├── .gitignore
├── CHANGELOG.md
├── LICENSE                       # MIT
├── README.md                     # English
└── README.ru.md                  # Русский
```

## Состав бандла

| Файл | Описание |
|------|----------|
| `model.xml` | Статический OpenVINO IR граф |
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

- [strokinkv/bge-m3-int8-ov](https://huggingface.co/strokinkv/bge-m3-int8-ov)
- [Оригинальная модель](https://huggingface.co/BAAI/bge-m3)
- [Статья BGE-M3](https://arxiv.org/abs/2402.03216)
- [ai2npu — NPU инференс](https://github.com/strokinkv/ai2npu)
