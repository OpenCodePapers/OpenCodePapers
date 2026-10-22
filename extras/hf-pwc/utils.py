"""
Helpers shared by the import scripts.
"""

import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

# ==================================================================================================

BENCHMARKS_DIR = "../../dataset/benchmarks/"
CATALOG_DIR = "data/catalog/"
OVERRIDES_PATH = "data/overrides.json"
REVIEWED_PATH = "data/reviewed_evaluations.json"

# Minimum size of new benchmarks
MIN_ENTRIES = 5
MIN_PAPERS = 3

# Leaderboard sources of external evaluations that are papers on the site, like
# "https://paperswithcode.co/paper/2603.26017" or "/paper/2606.07229"
PAPER_SOURCE_RE = re.compile(
    r"^(?:https?://(?:www\.)?paperswithcode\.co)?/paper/([^/?#]+)"
)
ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")

# ==================================================================================================


def normalize_title(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def arxiv_id_of_link(link: str) -> str:
    m = ARXIV_ID_RE.search(link or "")
    return m.group(1) if m else ""


def bench_dataset_part(title: str) -> str:
    """Dataset part of a benchmark title like "<task>-on-<dataset>"."""
    return title.rsplit("-on-", 1)[-1]


def compact_name(s: str) -> str:
    """Dataset name without separators and with short years ("AIME 2024" -> "aime24")."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"20(\d\d)", r"\1", s.lower()))


def leaderboard_key(evaluation: dict) -> str:
    return f"{evaluation['task_id']}/{evaluation['dataset_id']}"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data, compact: bool = False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1 if compact else 2, ensure_ascii=False)


def load_catalog():
    tasks = {t["id"]: t for t in load_json(os.path.join(CATALOG_DIR, "tasks.json"))}
    datasets = {
        d["id"]: d for d in load_json(os.path.join(CATALOG_DIR, "datasets.json"))
    }
    evaluations = load_json(os.path.join(CATALOG_DIR, "evaluations.json"))
    return tasks, datasets, evaluations


def load_benchmarks() -> Dict[str, dict]:
    benchmarks = {}
    for fname in sorted(os.listdir(BENCHMARKS_DIR)):
        if fname.endswith(".json"):
            data = load_json(os.path.join(BENCHMARKS_DIR, fname))
            benchmarks[data["title"]] = data
    return benchmarks


def load_overrides() -> dict:
    if not os.path.isfile(OVERRIDES_PATH):
        return {}
    return load_json(OVERRIDES_PATH)


def load_reviewed_evaluations() -> Dict[str, set]:
    """Ids of evaluations whose entries were reviewed: "accepted" and "excluded"."""
    if not os.path.isfile(REVIEWED_PATH):
        return {"accepted": set(), "excluded": set()}
    data = load_json(REVIEWED_PATH)
    return {k: set(data.get(k, [])) for k in ("accepted", "excluded")}


def write_reviewed_evaluations(reviewed: Dict[str, set]):
    """One line per list, the file holds thousands of ids."""
    lines = []
    for k in ("accepted", "excluded"):
        ids = sorted(reviewed[k], key=int)
        lines.append(f'  "{k}": {json.dumps(ids)}')
    with open(REVIEWED_PATH, "w", encoding="utf-8") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")


def task_chain(tasks: Dict[str, dict], task_id: str) -> List[str]:
    """Task names from the top-level task down to the given one."""
    chain = []
    task = tasks.get(task_id)
    while task and len(chain) < 10:
        chain.append(task["name"])
        task = tasks.get(task["parent_id"])
    return chain[::-1]


# ==================================================================================================


def is_importable(evaluation: dict) -> bool:
    """Results of the evaluated paper or of another paper on the site, not of websites."""
    if not evaluation.get("external"):
        return True
    return bool(PAPER_SOURCE_RE.match(evaluation.get("source_url") or ""))


def reporting_paper_ref(evaluation: dict) -> str:
    """Id (or arXiv id) of the paper an external result was taken from, else ""."""
    if not evaluation.get("external"):
        return ""
    m = PAPER_SOURCE_RE.match(evaluation.get("source_url") or "")
    return m.group(1) if m else ""


# ==================================================================================================


def clean_value(raw) -> str:
    """Metric value as a string, like the benchmark files store them, "" if not numeric."""
    if isinstance(raw, bool) or raw is None:
        return ""
    if isinstance(raw, (int, float)):
        text = repr(raw)
        if "e" in text or "E" in text:
            text = format(Decimal(text), "f")
        return text
    text = str(raw).strip()
    if re.fullmatch(r"[-+]?\d+,\d+", text):
        text = text.replace(",", ".")
    if not re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)%?", text):
        return ""
    if text.startswith("."):
        text = "0" + text
    return text


def metric_value(raw) -> Optional[float]:
    """Numeric value of a metric, like tests/test_benchmark_files.py reads it."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if "," in cleaned and "." not in cleaned:
        if re.match(r"^\s*[-+]?\d+,\d+(?:[eE][-+]?\d+)?\s*$", cleaned):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    return float(match.group(0)) if match else None


NUMERIC_CELL_RE = re.compile(
    r"\s*([-+]?\d*\.?\d+)\s*%?\s*(\(?\s*(±|\+-|\+/-)\s*\d*\.?\d+\s*%?\s*\)?)?\s*"
)


def _decimal(raw) -> Optional[Decimal]:
    """Number of a value like "89.6", "89.6±0.3" or "45.2%", None for "ResNet100"."""
    text = clean_value(raw) if not isinstance(raw, str) else raw.replace(",", "")
    m = NUMERIC_CELL_RE.fullmatch(text or "")
    if not m:
        return None
    try:
        return Decimal(m.group(1))
    except InvalidOperation:
        return None


def values_agree(hf_raw, ocp_raw, min_digits: int = 3) -> Optional[int]:
    """
    Factor (1, 100, or -100 for /100) by which the hf value equals the benchmark value
    after rounding to the lesser precision, None if they differ or have less than
    `min_digits` significant digits (like "75").
    """
    a, b = _decimal(hf_raw), _decimal(ocp_raw)
    if a is None or b is None:
        return None
    for factor in (1, 100, -100):
        x = a * factor if factor > 0 else a / 100
        exp = max(x.as_tuple().exponent, b.as_tuple().exponent)
        q = Decimal(1).scaleb(exp)
        xr, br = x.quantize(q), b.quantize(q)
        if xr != br or br == 0:
            continue
        digits = min(len(d.as_tuple().digits) for d in (a, b, br))
        return factor if digits >= min_digits else None
    return None


def scale_value(text: str, factor: int) -> str:
    """Convert a hf value by the factor of `values_agree`, keeping its precision."""
    if factor == 1 or not text:
        return text
    percent = text.endswith("%")
    d = Decimal(text.rstrip("%"))
    d = d * 100 if factor > 0 else d / 100
    exp = Decimal(text.rstrip("%")).as_tuple().exponent
    exp = exp + 2 if factor > 0 else exp - 2
    d = d.quantize(Decimal(1).scaleb(min(0, exp)))
    return format(d, "f") + ("%" if percent else "")
