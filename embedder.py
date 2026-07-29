"""embedder.py — эмбеддинги e5 напрямую через onnxruntime.

Зачем отдельный модуль: sentence-transformers безусловно импортирует torch,
а torch+MKL на CPU этого VPS (нет AVX) убивается ядром по SIGILL. Плюс связка
sentence-transformers/optimum дважды ломалась на обновлении версий: резолв
onnx/model.onnx при local_files_only возвращал None.

Здесь весь путь явный: tokenizers -> onnxruntime -> пулинг и нормализация в
numpy. Torch не импортируется вообще. Параметры (пулинг, длина, нормализация,
pad-токен) читаются из файлов самой модели, чтобы вектора совпадали с теми,
что уже лежат в БД.
"""
import json
import logging
import os
import threading

import numpy as np

import config as settings

log = logging.getLogger("gdelt_rss")

_ENGINE = None
_LOCK = threading.Lock()

# Потоки onnxruntime. Дефолт = все ядра, что на 4-ядерной машине с параллельным
# cron-прогоном даёт драку за CPU. Переопределяется через окружение.
_THREADS = int(os.environ.get("EMBED_THREADS", "2"))
_LONG_TEXT_CHARS = int(os.environ.get("EMBED_LONG_CHARS", "600"))
_LONG_TEXT_BATCH = int(os.environ.get("EMBED_LONG_BATCH", "4"))


def _model_dir():
    """Каталог снапшота модели в HF-кэше. Путь и имя модели — из settings."""
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        # Модуль могли импортировать в обход daily_collector.py (например из
        # apply_patches.py), тогда HF_HOME ещё не выставлен. Берём кэш рядом
        # с самим проектом — там же, куда его кладёт daily_collector.
        hf_home = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".cache", "huggingface")
        log.warning("embedder: HF_HOME не задан, беру %s", hf_home)
    repo = "models--" + settings.EMBEDDING_MODEL.replace("/", "--")
    snaps = os.path.join(hf_home, "hub", repo, "snapshots")
    if not os.path.isdir(snaps):
        raise FileNotFoundError("нет снапшотов модели: %s" % snaps)
    dirs = [os.path.join(snaps, d) for d in os.listdir(snaps)]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError("каталог снапшотов пуст: %s" % snaps)
    return max(dirs, key=os.path.getmtime)


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        if default is None:
            raise
        log.warning("embedder: не читается %s (%s), беру дефолт", path, exc)
        return default


class _Engine:
    """Совместим по вызову с SentenceTransformer.encode(), чтобы pipeline.py
    не пришлось переписывать."""

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = _model_dir()
        onnx_path = os.path.join(d, "onnx", "model.onnx")
        tok_path = os.path.join(d, "tokenizer.json")
        for p in (onnx_path, tok_path):
            if not os.path.exists(p):
                raise FileNotFoundError("нет файла модели: %s" % p)

        # Пулинг: берём из 1_Pooling/config.json. Если там не mean — падаем,
        # а не считаем молча по-другому: иначе новые вектора не сойдутся
        # со старыми в БД, и вся дедупликация поедет.
        pool = _read_json(os.path.join(d, "1_Pooling", "config.json"))
        if not pool.get("pooling_mode_mean_tokens"):
            raise RuntimeError("ожидался mean pooling, в конфиге: %r" % pool)
        dim = int(pool.get("word_embedding_dimension", settings.EMBEDDING_DIM))
        if dim != settings.EMBEDDING_DIM:
            raise RuntimeError("размерность модели %d != settings.EMBEDDING_DIM %d"
                               % (dim, settings.EMBEDDING_DIM))

        # Нормализация — только если она есть в конвейере модели.
        mods = _read_json(os.path.join(d, "modules.json"), default=[])
        self.normalize = any("Normalize" in m.get("type", "") for m in mods)

        max_seq = int(_read_json(os.path.join(d, "sentence_bert_config.json"),
                                 default={}).get("max_seq_length", 512))

        so = ort.SessionOptions()
        so.intra_op_num_threads = _THREADS
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Арена-аллокатор ORT кэширует буферы активаций и память назад не
        # отдаёт: на батче 16 x 512 токенов она разрасталась до ~1.4 ГБ поверх
        # весов. На 8-гигабайтной машине с параллельным cron это дороже, чем
        # выигрыш в скорости, поэтому арену выключаем.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(onnx_path, so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

        self.tok = Tokenizer.from_file(tok_path)
        self.tok.enable_truncation(max_length=max_seq)
        spec = _read_json(os.path.join(d, "special_tokens_map.json"), default={})
        pad_tok = spec.get("pad_token", "<pad>")
        if isinstance(pad_tok, dict):
            pad_tok = pad_tok.get("content", "<pad>")
        pad_id = self.tok.token_to_id(pad_tok)
        if pad_id is None:
            raise RuntimeError("не найден pad-токен %r в токенизаторе" % pad_tok)
        self.tok.enable_padding(pad_id=pad_id, pad_token=pad_tok)

        log.info("embedder: onnxruntime, %d потока, max_seq=%d, normalize=%s",
                 _THREADS, max_seq, self.normalize)

    def _forward(self, chunk):
        encs = self.tok.encode_batch(chunk)
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(["last_hidden_state"], feed)[0]

        # Mean pooling по attention-маске — ровно как в sentence_transformers
        # Pooling: sum(h * mask) / clamp(sum(mask), min=1e-9).
        m = mask[:, :, None].astype(np.float32)
        vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        if self.normalize:
            vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True),
                                1e-12, None)
        return vec.astype(np.float32)

    def encode(self, texts, batch_size=16, **_ignored):
        """Лишние kwargs (convert_to_numpy, show_progress_bar) глотаем —
        вызывающий код писался под SentenceTransformer."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, settings.EMBEDDING_DIM), np.float32)
        batch_size = max(1, int(batch_size or 16))
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            # Длинные тексты режем на подбатчи: пиковая память ORT растёт как
            # batch * seq_len, и тела статей на 2000 знаков дают активации
            # на порядок больше, чем заголовки.
            if max(len(t) for t in chunk) > _LONG_TEXT_CHARS and len(chunk) > _LONG_TEXT_BATCH:
                for j in range(0, len(chunk), _LONG_TEXT_BATCH):
                    out.append(self._forward(chunk[j:j + _LONG_TEXT_BATCH]))
                continue
            try:
                out.append(self._forward(chunk))
            except Exception:
                # Батч мог упасть на одном кривом тексте. Пробуем поштучно,
                # чтобы не потерять остальные; безнадёжные — нулевой вектор.
                log.exception("embedder: батч %d-%d упал, иду поштучно",
                              i, i + len(chunk))
                for t in chunk:
                    try:
                        out.append(self._forward([t]))
                    except Exception:
                        log.exception("embedder: текст пропущен (%.60s)", t)
                        out.append(np.zeros((1, settings.EMBEDDING_DIM), np.float32))
        return np.vstack(out)


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                log.info("Загрузка e5-модели %s (onnxruntime) ...",
                         settings.EMBEDDING_MODEL)
                _ENGINE = _Engine()
    return _ENGINE
