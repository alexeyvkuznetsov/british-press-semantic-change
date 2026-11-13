"""Все параметры проекта: пути, десятилетия, слова и оси.

Списки слов меняются только здесь. После добавления новых слов нужно
перезапустить 00_prepare_data.ipynb, затем аналитические ноутбуки.
"""
import os
from pathlib import Path

# --- Пути -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Каталог моделей 2023 года; можно указать LWM_MODEL_DIR_2023 до импорта.
# Ищем ранее скачанные файлы по именам *.model.
_MODEL_NAMES = [f"{period}.model" for period in
                ["1800s", "1810s", "1820s", "1830s", "1840s", "1850s",
                 "1860s", "1870s", "1880s", "1890s", "1900s", "1910s"]]
_MODEL_DIR_OVERRIDE = os.environ.get("LWM_MODEL_DIR_2023")
_MODEL_DIR_CANDIDATES = [ROOT / "data" / "models_2023", ROOT / "data" / "models", ROOT / "data",
                         ROOT / "models", ROOT]
if _MODEL_DIR_OVERRIDE:
    MODEL_DIR = Path(_MODEL_DIR_OVERRIDE).expanduser().resolve()
else:
    MODEL_DIR = max(_MODEL_DIR_CANDIDATES,
                    key=lambda d: sum((d / name).is_file() for name in _MODEL_NAMES))
CACHE_DIR = ROOT / "data" / "cache_2023"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"

# --- Данные ---------------------------------------------------------------
ZENODO_RECORD = 7887305  # Pedrazzini 2023, невыровненные модели
PERIODS = ["1800s", "1810s", "1820s", "1830s", "1840s", "1850s",
           "1860s", "1870s", "1880s", "1890s", "1900s", "1910s"]
MODEL_DIM = 200
MODEL_SUFFIXES = [".model", ".model.wv.vectors.npy", ".model.syn1neg.npy"]
TOPN = 500  # расширенный список кандидатов, без фильтрации
REPORT_TOPN = 30  # объём списка для содержательного анализа
OCR_RULES = ROOT / "ocr_exclusions.csv"

# --- Слова, для которых сохраняются ближайшие соседи -----------------------
NEIGHBOR_TARGETS = ["imperialism", "imperialist", "imperialists", "imperial", "empire",
                    "emperor", "empress", "jingoism", "jingo", "czar", "russia", "russian",
                    "russophobia", "autocracy", "tsar"]

# Слова, отсутствие которых в части десятилетий упоминается в статье
PROBE_WORDS = ["imperialism", "imperialist", "imperialists", "jingo", "jingoism",
               "russophobia", "autocracy"]

# --- Баланс «своё – чужое» (раздел 3 статьи) -------------------------------
# Измерение по двум наборам слов: направление между их центрами (An et al. 2018),
# положение слова — показатель ассоциации s(w, A, B) (Caliskan et al. 2017),
# в предыдущей работе автора названный балансом семантических полей. «Своё» — слова, обозначающие
# британскую монархию и её владения, «чужое» — иностранные державы и их правители
# (Koebner & Schmidt 1964 о «чужих» империях середины века). Наборы фиксированы для всех десятилетий; наличие проверяется. Базовая линия — контрольное условие (Dubossarsky et al. 2017).
BALANCE_TARGETS = ["imperialism", "imperialist", "imperialists", "imperial", "empire", "emperor", "empress", "jingoism", "jingo"]
OWN = ["british", "britain", "colonies", "colonial", "india", "canada", "dominion", "queen"]
FOREIGN = ["france", "french", "napoleon", "bonaparte", "austria", "austrian",
           "russia", "russian", "turkey", "ottoman", "germany", "german"]
OWN_GEOGRAPHIC = ["britain", "british", "england", "english"]
FOREIGN_GEOGRAPHIC = ["france", "french", "austria", "austrian", "russia", "russian",
                      "turkey", "turkish", "germany", "german"]
BASE = ["kingdom", "government", "army", "nation", "crown"]  # базовая линия дрейфа

# --- Семантические оси (раздел 4 статьи) ----------------------------------
# Пары антонимов (отрицательный полюс, положительный полюс), ось — среднее направлений внутри пар (Kozlowski et al. 2019).
# Полюса выбраны по историографии образа России в британской печати:
# тирания и варварство (Некрасова, Барская 2023; Митина 2024).
# Надёжность пар проверяется по Boutyline & Johnston 2026 (ноутбук 02).
AXES = {
    "liberty": {
        "label": "деспотизм – свобода",
        "pairs": [("despotism", "liberty"), ("tyranny", "freedom"),
                  ("despotic", "constitutional"), ("tyrannical", "free"),
                  ("oppression", "emancipation"), ("servitude", "independence"),
                  ("arbitrary", "lawful"), ("oppressive", "liberal"),
                  ("enslaved", "independent"), ("persecution", "toleration")],
    },
    "civilisation": {
        "label": "варварство – цивилизация",
        "pairs": [("barbarous", "civilised"), ("barbarism", "civilisation"),
                  ("uncivilized", "civilized"), ("savage", "enlightened"),
                  ("barbarity", "humanity"), ("brutal", "humane"),
                  ("rude", "refined"), ("backward", "progressive"),
                  ("ferocious", "gentle"), ("cruel", "merciful"),
                  ("superstition", "knowledge"), ("fanatical", "rational"),
                  ("primitive", "advanced")],
    },
    "east_west": {
        "label": "восток – запад",
        "pairs": [("asiatic", "european"), ("eastern", "western"), ("asia", "europe")],
    },
}
# Решения по итогам ноутбука 02 (диагностика параллельности)
DROP_PAIRS = {
    "civilisation": [("primitive", "advanced"), ("backward", "progressive"),
                     ("uncivilized", "civilized")],
}
WORKING_AXES = ["liberty", "civilisation"]  # ось «восток – запад» признана ненадёжной

# --- Россия и другие страны ------------------------------------------------
RUSSIA = "russia"
COUNTRIES = ["russia", "france", "austria", "prussia", "germany", "turkey",
             "spain", "italy", "england", "britain", "america"]
COUNTRY_LABELS = {"russia": "Россия", "france": "Франция", "austria": "Австрия",
                  "prussia": "Пруссия", "germany": "Германия", "turkey": "Турция",
                  "spain": "Испания", "italy": "Италия", "england": "Англия",
                  "britain": "Британия", "america": "Америка"}
# Подписи событий на рис. 2. Это ориентиры для сопоставления, а не результаты модели.
EVENTS = {"1830s": "Польское\nвосстание", "1850s": "Крымская\nвойна",
          "1870s": "Восточный\nкризис", "1900s": "1905 / 1907 гг.", "1910s": "1914 г."}

# Сценарии чувствительности сравнительной группы (применяются ко всем периодам).
# Основная группа не меняется. Не объединяем исторически разные обозначения.
CONTROL_EXCLUSIONS = {
    "без Турции": ["turkey"],
    "без Германии и Пруссии": ["germany", "prussia"],
    "без Англии и Пруссии": ["england", "prussia"],
}

# --- Воспроизводимость и оформление ----------------------------------------
PALETTE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
           "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7",
           "grey": "#999999"}
FIG_DPI = 300
FIG_WIDTH_CM = 16


def all_words():
    """Все слова, векторы которых нужно сохранить в кэш."""
    words = set(NEIGHBOR_TARGETS + PROBE_WORDS + BALANCE_TARGETS + OWN + FOREIGN + OWN_GEOGRAPHIC + FOREIGN_GEOGRAPHIC + BASE + COUNTRIES)
    for ax in AXES.values():
        for a, z in ax["pairs"]:
            words.update([a, z])
    return sorted(words)


def working_pairs(axis):
    """Пары рабочей оси после исключения слабых пар."""
    drop = set(DROP_PAIRS.get(axis, []))
    return [p for p in AXES[axis]["pairs"] if p not in drop]
