"""Общие функции: скачивание моделей, кэш, меры и оформление результатов."""
import gc
import hashlib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config as C

# ============================================================================
# Скачивание моделей
# ============================================================================

def zenodo_files(record=C.ZENODO_RECORD):
    """Список файлов записи Zenodo: имя, размер, md5, ссылка."""
    import requests
    r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json()["files"]:
        out.append({"key": f["key"], "size": f["size"],
                    "md5": f["checksum"].split(":")[-1],
                    "url": f["links"].get("content") or f"https://zenodo.org/records/{record}/files/{f['key']}?download=1"})
    return out


def md5sum(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def local_model_status(periods=C.PERIODS, model_dir=C.MODEL_DIR):
    """Наличие всех трёх файлов каждого десятилетия."""
    rows = []
    for period in periods:
        for suffix in C.MODEL_SUFFIXES:
            path = model_dir / (period + suffix)
            rows.append({"десятилетие": period, "файл": path.name,
                         "найден": "да" if path.is_file() else "нет",
                         "размер, МБ": round(path.stat().st_size / 2**20, 1) if path.is_file() else None,
                         "путь": str(path)})
    return pd.DataFrame(rows)


def download_models(periods=C.PERIODS, model_dir=C.MODEL_DIR, verify_md5=True):
    """Загрузка только недостающих или повреждённых файлов (Zenodo 7887305)."""
    import requests
    import shutil
    from tqdm.auto import tqdm
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    model_dir.mkdir(parents=True, exist_ok=True)
    files = {f["key"]: f for f in zenodo_files()}
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2,
                   status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
    report = []
    for p in periods:
        for suffix in C.MODEL_SUFFIXES:
            info = files[p + suffix]
            path = model_dir / info["key"]
            exists = path.is_file() and path.stat().st_size == info["size"]
            if exists and (not verify_md5 or md5sum(path) == info["md5"]):
                status = "уже есть; проверен"
            else:
                status = "скачан заново" if path.exists() else "скачан"
                if shutil.disk_usage(model_dir).free < info["size"] + 100 * 2**20:
                    raise OSError(f"Недостаточно места для {path.name}")
                tmp = path.with_suffix(path.suffix + ".part")
                try:
                    url = info["url"]
                    # API links.self может вести к метаданным; content — к файлу.
                    with session.get(url, stream=True, timeout=(30, 600)) as r:
                        r.raise_for_status()
                        with open(tmp, "wb") as f, tqdm(total=info["size"], unit="B", unit_scale=True, desc=path.name) as bar:
                            for block in r.iter_content(chunk_size=4 << 20):
                                if block:
                                    f.write(block); bar.update(len(block))
                    if tmp.stat().st_size != info["size"] or (verify_md5 and md5sum(tmp) != info["md5"]):
                        raise IOError(f"Размер или MD5 не совпадает: {path.name}")
                    tmp.replace(path)
                finally:
                    if tmp.exists(): tmp.unlink()
            report.append({"десятилетие": p, "файл": path.name, "размер, МБ": round(info["size"] / 2**20, 1),
                           "статус": status, "проверка": "в порядке"})
    session.close()
    return pd.DataFrame(report)


# ============================================================================
# Кэш векторов, соседей и частот
# ============================================================================
CACHE_FILE = C.CACHE_DIR / "cache.pkl"
CACHE_VERSION = 2


def load_model(period):
    from gensim.models import Word2Vec
    model = Word2Vec.load(str(C.MODEL_DIR / f"{period}.model"), mmap="r")
    if model.wv.vector_size != C.MODEL_DIM:
        raise ValueError(f"{period}: размерность {model.wv.vector_size}, ожидалось {C.MODEL_DIM}")
    return model


def build_cache(words=None, neighbor_targets=None, topn=C.TOPN):
    """Один проход по моделям. Кэш содержит только слова проекта, не полные словари."""
    from tqdm.auto import tqdm
    words = sorted(set(C.all_words() if words is None else words))
    neighbor_targets = C.NEIGHBOR_TARGETS if neighbor_targets is None else neighbor_targets
    words = sorted(set(words) | set(neighbor_targets))
    cache = {"record": C.ZENODO_RECORD, "version": CACHE_VERSION, "topn": topn,
             "neighbor_targets": list(neighbor_targets), "vocab": {}, "vocab_size": {},
             "vectors": {}, "neighbors": {}, "words": words, "counts": {},
             "corpus_total_words": {}, "retained_tokens": {}, "model_parameters": {}}
    for p in tqdm(C.PERIODS, desc="Модели"):
        model = load_model(p); m = model.wv
        cache["model_parameters"][p] = {
            "period": p, "vector_size": m.vector_size, "vocab_size": len(m),
            **{k: getattr(model, k, None) for k in
               ["sg", "window", "min_count", "epochs", "negative", "corpus_total_words", "corpus_count"]},
        }
        m.fill_norms(force=True)
        present = {w for w in words if w in m.key_to_index}
        cache["vocab"][p] = present
        cache["vocab_size"][p] = len(m)
        cache["vectors"][p] = {w: m.get_vector(w, norm=True).copy() for w in present}
        cache["neighbors"][p] = {w: m.most_similar(w, topn=topn) for w in neighbor_targets if w in present}
        # count — число употреблений, сохранённое Gensim при обучении; не псевдочастота.
        counts = m.expandos.get("count")
        cache["counts"][p] = {w: int(m.get_vecattr(w, "count")) for w in present} if counts is not None else {}
        total = getattr(model, "corpus_total_words", None)
        cache["corpus_total_words"][p] = int(total) if total is not None and total > 0 else None
        cache["retained_tokens"][p] = int(np.sum(counts, dtype=np.int64)) if counts is not None else None
        del counts, m, model
        gc.collect()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    with open(tmp, "wb") as f: pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_FILE)
    return cache


def load_cache():
    if not CACHE_FILE.exists():
        raise FileNotFoundError("Кэш не найден. Сначала выполните 00_prepare_data.ipynb.")
    with open(CACHE_FILE, "rb") as f: cache = pickle.load(f)
    if cache.get("record") != C.ZENODO_RECORD or cache.get("version") != CACHE_VERSION:
        raise ValueError("Кэш другого набора или формата. Перезапустите 00_prepare_data.ipynb.")
    missing = set(C.all_words()) - set(cache["words"])
    if missing or not set(C.NEIGHBOR_TARGETS).issubset(cache["neighbor_targets"]) or cache["topn"] < C.TOPN:
        raise ValueError(f"Настройки изменились; перестройте кэш. Новые слова: {sorted(missing)}")
    return cache


def vocabulary_table(cache, words):
    rows = {w: ["+" if w in cache["vocab"][p] else "–" for p in C.PERIODS] for w in words}
    t = pd.DataFrame(rows, index=decade_labels()).T
    t["всего"] = [sum(w in cache["vocab"][p] for p in C.PERIODS) for w in words]
    return t


def frequency_table(cache, words):
    """Пропуск — OOV или отсутствующие метаданные, не нулевая частота.
    Частота на миллион рассчитывается по corpus_total_words (если доступно).
    Доля среди сохранённых токенов приводится отдельно с другим знаменателем.
    """
    rows = []
    for p, lab in zip(C.PERIODS, decade_labels()):
        total = cache["corpus_total_words"][p]; retained = cache["retained_tokens"][p]
        for w in words:
            n = cache["counts"][p].get(w)
            rows.append({"десятилетие": lab, "слово": w, "в_словаре": w in cache["vocab"][p],
                         "count": n, "corpus_total_words": total, "retained_tokens": retained,
                         "per_million_corpus": 1e6*n/total if n is not None and total else np.nan,
                         "per_million_retained": 1e6*n/retained if n is not None and retained else np.nan})
    return pd.DataFrame(rows)


def require_words(V, words, context):
    """Не менять состав опорных наборов молча при отсутствии слова."""
    missing = {p: sorted(set(words) - set(V[p])) for p in C.PERIODS}
    missing = {p: w for p, w in missing.items() if w}
    if missing: raise ValueError(f"{context}: отсутствуют обязательные слова {missing}. См. 01_vocabulary_project_words.csv; не меняйте наборы без явного решения.")


# ============================================================================
# Семантические оси (Kozlowski et al. 2019; Boutyline & Johnston 2026)
# ============================================================================

def offsets(V, pairs):
    """Нормированные векторы разностей z − a для пар (a, z)."""
    o = np.array([V[z] - V[a] for a, z in pairs])
    norms = np.linalg.norm(o, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Нулевое или некорректное направление опорной пары")
    return o / norms


def parallelism(V, pairs):
    """Средний косинус между разностями пар (Boutyline & Johnston: параллельность)."""
    o = offsets(V, pairs)
    s = o @ o.T
    n = len(o)
    return float(s[~np.eye(n, dtype=bool)].mean()) if n >= 2 else np.nan


def pair_parallelism(V, pairs):
    """Параллельность каждой пары с остальными парами оси."""
    o = offsets(V, pairs)
    s = o @ o.T
    n = len(o)
    return (s.sum(axis=1) - np.diag(s)) / (n - 1) if n >= 2 else np.full(n, np.nan)


def axis_vector(V, pairs):
    """Семантическая ось: нормированное среднее направлений внутри пар (Kozlowski et al. 2019)."""
    v = offsets(V, pairs).mean(axis=0)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Невозможно нормировать семантическую ось")
    return v / norm


def difference_from_controls(V, pairs, target, controls):
    """Проекция слова-мишени на ось минус средняя проекция контрольных слов."""
    a = axis_vector(V, pairs)
    return float(V[target] @ a - np.mean([V[c] @ a for c in controls]))


def available_pairs(V, pairs):
    """Только для диагностики: пары, оба слова которых доступны."""
    return [(a, z) for a, z in pairs if a in V and z in V]


def diagnostic_parallelism(V, pairs):
    """Сохраняет исходный порядок пар; для недоступных пар возвращает NaN."""
    present = available_pairs(V, pairs)
    scores = dict(zip(present, pair_parallelism(V, present))) if len(present) >= 2 else {}
    return [scores.get(pair, np.nan) for pair in pairs]


def read_ocr_rules(path=C.OCR_RULES):
    """Ручные правила точного совпадения: * = все периоды/целевые слова."""
    rules = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["period", "target", "neighbor", "reason"]
    if list(rules.columns) != required:
        raise ValueError(f"Ожидаемые столбцы OCR: {required}")
    for row in rules.itertuples(index=False):
        if row.period not in ["*"] + C.PERIODS or row.target not in ["*"] + C.NEIGHBOR_TARGETS:
            raise ValueError(f"Неверная область правила OCR: {row}")
        if not row.neighbor or not row.reason.strip():
            raise ValueError(f"Нужно указать слово и причину: {row}")
    if rules.duplicated(["period", "target", "neighbor"]).any():
        raise ValueError("Повторяющиеся правила OCR")
    return rules


def neighbor_review(cache, rules):
    """Все кандидаты с исходным рангом; исключения не меняют векторы."""
    lookup = {(r.period, r.target, r.neighbor): r.reason for r in rules.itertuples(index=False)}
    rows = []
    for p in C.PERIODS:
        for w in C.NEIGHBOR_TARGETS:
            rank = 0
            for raw_rank, (neighbor, sim) in enumerate(cache["neighbors"][p].get(w, []), 1):
                reason = next((lookup[k] for k in [(p,w,neighbor), (p,"*",neighbor),
                                                  ("*",w,neighbor), ("*","*",neighbor)] if k in lookup), "")
                if not reason:
                    rank += 1
                rows.append({"period": p, "target": w, "raw_rank": raw_rank,
                             "neighbor": neighbor, "cosine": float(sim), "excluded": bool(reason),
                             "reason": reason, "filtered_rank": rank if not reason else None})
    return pd.DataFrame(rows, columns=["period", "target", "raw_rank", "neighbor", "cosine",
                                       "excluded", "reason", "filtered_rank"])


# ============================================================================
# Баланс полей (как в первой статье: разность средних сходств)
# ============================================================================

def field_balance(V, word, positive, negative):
    """Баланс семантических полей: среднее сходство с первым набором минус среднее со вторым."""
    pos = [w for w in positive if w != word]
    neg = [w for w in negative if w != word]
    return (np.mean([V[word] @ V[w] for w in pos])
            - np.mean([V[word] @ V[w] for w in neg]))


# ============================================================================
# Оформление и сохранение
# ============================================================================

def decade_labels(periods=C.PERIODS):
    return [f"{p[:4]}-е" for p in periods]


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11,
        "figure.dpi": 110, "savefig.dpi": C.FIG_DPI, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
        "axes.grid.axis": "y",
    })


COMMA = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",").replace("-", "−"))


def figure(height_cm=8, width_cm=C.FIG_WIDTH_CM, **kw):
    return plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54), **kw)


def save_fig(fig, name):
    """Сохраняет рисунок в PNG (300 dpi)."""
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png")
    print(f"Сохранено: results/figures/{name}.png")


def save_table(df, name, index=True):
    """Сохраняет таблицу в CSV (UTF-8 с BOM, открывается в Excel без перекодировки)."""
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB_DIR / f"{name}.csv", index=index, encoding="utf-8-sig")
    print(f"Сохранено: results/tables/{name}.csv")
