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
optimum-cli export openvino --model BAAI/bge-m3 --task feature-extraction --weight-format int8 models/bge-m3-int8-ov
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
model = core.compile_model("models/bge-m3-int8-ov/openvino_model.xml", "CPU")
tokenizer = AutoTokenizer.from_pretrained("models/bge-m3-int8-ov")

text = "Что такое BGE M3?"
encoded = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
result = model({"input_ids": encoded["input_ids"].astype(np.int64), "attention_mask": encoded["attention_mask"].astype(np.int64)})
last_hidden = result["last_hidden_state"]
embedding = last_hidden[0, 0, :]  # CLS-токен = эмбеддинг предложения, shape: [1024]
```

## Параметры модели

| Свойство | Значение |
|----------|----------|
| Целевая модель | [strokinkv/bge-m3-int8-ov](https://huggingface.co/strokinkv/bge-m3-int8-ov) |
| Базовая модель | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Лицензия | MIT |
| Входная форма | динамическая |
| Входы | `input_ids` (int64), `attention_mask` (int64) |
| Выход | `last_hidden_state` [batch, seq, 1024] float32 |
| Эмбеддинг | CLS-токен: `last_hidden_state[:, 0, :]` → [1024] |
| Квантизация | INT8 asymmetric (NNCF) |
| Размерность | 1024 |
| Макс. длина | 8192 токена (ограничение оригинальной модели) |

## Структура проекта

```
bge-m3-openvino/
├── scripts/
│   ├── test_model.py             # Валидация модели
│   └── requirements.txt
├── .github/workflows/
│   └── publish.yml               # CI/CD
├── .gitignore
├── LICENSE                       # MIT
├── README.md                     # English
└── README.ru.md                  # Русский
```

## Состав бандла

| Файл | Описание |
|------|----------|
| `openvino_model.xml` | OpenVINO IR граф |
| `openvino_model.bin` | OpenVINO IR веса (INT8, ~543 MB) |
| `tokenizer.json` | Токенизатор Hugging Face |
| `tokenizer_config.json` | Конфигурация токенизатора |
| `sentencepiece.bpe.model` | SentencePiece модель |
| `special_tokens_map.json` | Специальные токены |
| `config.json` | Конфигурация XLM-RoBERTa |

## Сигнатура модели

| Направление | Имя | Форма | Тип |
|-------------|-----|-------|-----|
| Вход | `input_ids` | [batch, seq] | int64 |
| Вход | `attention_mask` | [batch, seq] | int64 |
| Выход | `last_hidden_state` | [batch, seq, 1024] | float32 |

## Лицензия

MIT. Оригинальная модель — [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), Beijing Academy of Artificial Intelligence.

## Ссылки

- [strokinkv/bge-m3-int8-ov](https://huggingface.co/strokinkv/bge-m3-int8-ov)
- [Оригинальная модель](https://huggingface.co/BAAI/bge-m3)
- [Статья BGE-M3](https://arxiv.org/abs/2402.03216)
- [ai2npu — NPU инференс](https://github.com/strokinkv/ai2npu)
