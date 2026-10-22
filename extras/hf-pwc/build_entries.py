"""
Convert the evaluations of matched and new leaderboards to benchmark entries. Reviewed
entries are written to data/updates/, the others to data/review/ (entries_*, new_*,
new_columns.json).

- Results are credited to the evaluated model's paper, external results to the paper
  that reports them. Papers already in the dataset keep their name, link, code and date.
- The model name gets the evaluation harness and "(extra data)" if marked.
- Entries of papers already in the benchmark are skipped (same model name or value, or
  other variants of the paper).
- Metrics without a column become new columns if most new entries report them and the
  column is approved in data/overrides.json.
- New benchmarks need 5+ entries from 3+ papers and a code link. Leaderboards of the
  same dataset are combined, probable duplicates of existing benchmarks are skipped.
- Split leaderboards add the entries of their groups to the group's benchmark, groups of
  new benchmarks with the same title are combined.
"""

import collections
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import fetch_leaderboard_info
import fetch_paper_details
import hfpwc_utils as utils

sys.path.insert(0, "../../tests/")
from test_benchmark_files import (  # noqa: E402 pylint: disable=wrong-import-position,wrong-import-order
    check_malicious_injections,
    helper_link_validity,
)

# ==================================================================================================

MATCHES_PATH = "data/matches.json"
UPDATES_DIR = "data/updates/"
REVIEW_BATCH_ENTRIES = 300
MAX_REVIEW_ENTRIES = 150

MAX_METRICS = 10

NO_HARNESS = {"", "none", "not reported", "n/a", "unknown", "null"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LOWER_IS_BETTER_RE = re.compile(
    r"error|loss|\bwer\b|\bcer\b|\bmse\b|\bmae\b|rmse|\bfid\b|\bfvd\b|lpips|\bepe\b|\bece\b|"
    r"perplexity|\bppl\b|latency|\beer\b|absrel|abs rel|chamfer|mpjpe|\bnll\b|\bbpd\b|\bbpc\b|"
    r"\bkid\b|distance|\bade\b|\bfde\b|miss rate|collision|hausdorff|\bmce\b",
    re.IGNORECASE,
)

FIELD_P = {"key": "p", "label": "Paper"}
FIELD_C = {"key": "c", "label": "Code"}
FIELD_N = {"key": "n", "label": "ModelName"}
FIELD_D = {"key": "d", "label": "ReleaseDate", "sortable": "true"}

# ==================================================================================================


def clean_text(s: Optional[str]) -> str:
    return " ".join((s or "").split())


def valid_link(link: Optional[str]) -> bool:
    return bool(link) and not helper_link_validity(link)


def arxiv_abs(arxiv_id: Optional[str]) -> str:
    if not arxiv_id:
        return ""
    return "https://arxiv.org/abs/" + re.sub(r"v\d+$", "", arxiv_id)


def entry_key(entry: dict) -> Tuple[str, str]:
    return (
        utils.normalize_title(entry["p"]["name"]),
        utils.normalize_title(entry["n"]),
    )


def entry_id(entry: dict) -> str:
    return "+".join(entry["ids"])


def item_of(entry: dict, metrics: dict) -> dict:
    return {
        "p": entry["p"],
        "c": entry["c"],
        "n": entry["n"],
        "d": entry["d"],
        **{k: metrics[k] for k in sorted(metrics, key=lambda k: int(k[1:]))},
    }


class KnownPapers:
    """Name, link, code and date of the papers already in the dataset."""

    def __init__(self, benchmarks: Dict[str, dict]):
        infos = collections.defaultdict(
            lambda: collections.defaultdict(collections.Counter)
        )
        for data in benchmarks.values():
            for item in data["benchmark"]["items"]:
                p = item.get("p")
                if not isinstance(p, dict) or not p.get("name") or not p.get("link"):
                    continue
                keys = [("t", utils.normalize_title(p["name"]))]
                arxiv = utils.arxiv_id_of_link(p["link"])
                if arxiv:
                    keys.append(("a", arxiv))
                for key in keys:
                    infos[key]["paper"][(p["name"], p["link"])] += 1
                    if item.get("c"):
                        infos[key]["code"][item["c"]] += 1
                    if item.get("d"):
                        infos[key]["date"][item["d"]] += 1
        self.infos = infos

    def get(self, title: str, arxiv: Optional[str]) -> Optional[dict]:
        info = None
        if arxiv:
            info = self.infos.get(("a", re.sub(r"v\d+$", "", arxiv)))
        info = info or self.infos.get(("t", utils.normalize_title(title)))
        if not info:
            return None
        best = {k: c.most_common(1)[0][0] for k, c in info.items() if c}
        name, link = best["paper"]
        return {
            "name": name,
            "link": link,
            "code": best.get("code", ""),
            "date": best.get("date", ""),
        }


def load_paper_details(refs) -> Dict[str, dict]:
    details = {}
    for ref in refs:
        path = fetch_paper_details.paper_path(ref)
        if os.path.isfile(path):
            details[ref] = utils.load_json(path)
    return details


def official_repo(details: dict) -> str:
    for repo in details.get("repositories") or []:
        if repo.get("is_official") and valid_link(repo.get("url")):
            return repo["url"]
    return ""


def model_name(e: dict) -> str:
    name = clean_text(e.get("model_name"))
    harness = clean_text(e.get("harness"))
    if harness.lower() not in NO_HARNESS and harness.lower() not in name.lower():
        name += f" ({harness})"
    if e.get("uses_additional_data") and "extra" not in name.lower():
        name += " (extra data)"
    return name


def make_entry(
    e: dict, known: KnownPapers, details: Dict[str, dict]
) -> Tuple[dict, str]:
    """Benchmark entry of an evaluation (with its raw metrics), or the reason to skip it."""
    ref = utils.reporting_paper_ref(e)
    leaderboard = utils.leaderboard_source(e)
    if leaderboard:
        # credited to the leaderboard, dated by the model's release (else the evaluation)
        model_paper = known.get(e["paper_title"], e.get("paper_arxiv_id")) or {}
        title, arxiv = leaderboard[0], None
        date = (
            model_paper.get("date")
            or e["paper_published_date"]
            or e.get("evaluated_on")
        )
        code = model_paper.get("code") or e.get("code_url") or ""
        det = {"url_abs": leaderboard[1]}
    elif ref:
        det = details.get(ref) or {}
        if not det.get("title"):
            return {}, "reporting paper unknown"
        title, arxiv, date = det["title"], det.get("arxiv_id"), det.get("published")
        code = official_repo(det)
    else:
        det = details.get(e["paper_id"]) or {}
        title, arxiv, date = (
            e["paper_title"],
            e.get("paper_arxiv_id"),
            e["paper_published_date"],
        )
        code = e.get("code_url") or ""
    links = [
        det.get("url_abs") if leaderboard else "",
        arxiv_abs(arxiv),
        det.get("url_abs"),
        det.get("conference_url_abs"),
        det.get("source_url"),
        f"https://paperswithcode.co/paper/{det.get('id') or e['paper_id']}",
    ]

    paper = None if leaderboard else known.get(title, arxiv)
    name = paper["name"] if paper else clean_text(title)
    link = paper["link"] if paper else next((lk for lk in links if valid_link(lk)), "")
    code = (paper or {}).get("code") or (code if valid_link(code) else "")
    date = (paper or {}).get("date") or (date or "")[:10]
    if not name or not link:
        return {}, "no paper link"

    entry = {
        "p": {"name": name, "link": link},
        "c": code,
        "n": model_name(e),
        "d": date if DATE_RE.match(date) else "",
        "metrics": {},
        "ids": [e["id"]],
        "evaluated_paper": utils.normalize_title(e["paper_title"]),
        "website": leaderboard[0] if leaderboard else "",
        "methodology": clean_text(e.get("methodology"))[:160],
    }
    if not entry["n"]:
        return {}, "no model name"
    for metric, raw in e["metrics"].items():
        value = utils.clean_value(raw)
        if value:
            entry["metrics"][clean_text(metric)] = value
    if check_malicious_injections(item_of(entry, {}), ""):
        return {}, "suspicious text"
    return entry, ""


def combine(entries: List[dict], skipped: list) -> List[dict]:
    """Entries of the same paper and model as one, dropped if their values disagree."""
    groups = collections.defaultdict(list)
    for entry in entries:
        groups[entry_key(entry)].append(entry)
    result = []
    for group in groups.values():
        merged = dict(group[0])
        for k in ("metrics", "mapped", "extra"):
            if k in merged:
                merged[k] = dict(merged[k])
        merged["ids"] = sorted({i for e in group for i in e["ids"]}, key=int)
        conflict = False
        for other in group[1:]:
            for k in ("metrics", "mapped", "extra"):
                for m, v in other.get(k, {}).items():
                    if merged[k].get(m, v) != v:
                        conflict = True
                    merged[k].setdefault(m, v)
        if conflict:
            for entry in group:
                skipped.append(
                    skip_info(entry, "same paper and model with other values")
                )
            continue
        result.append(merged)

    # a model's result from its own paper is preferred over a leaderboard's one
    by_paper = {
        (e["evaluated_paper"], utils.normalize_title(e["n"]))
        for e in result
        if not e["website"]
    }
    kept = []
    for e in result:
        if (
            e["website"]
            and (e["evaluated_paper"], utils.normalize_title(e["n"])) in by_paper
        ):
            skipped.append(skip_info(e, "model already reported by its paper"))
            continue
        kept.append(e)
    return kept


def skip_info(entry: dict, reason: str, target: str = "") -> dict:
    return {
        "reason": reason,
        "target": target,
        "ids": entry.get("ids"),
        "paper": entry.get("p", {}).get("name"),
        "model": entry.get("n"),
        "metrics": entry.get("metrics"),
    }


def review_status(entry: dict, reviewed: Dict[str, set]) -> str:
    if all(i in reviewed["accepted"] for i in entry["ids"]):
        return "accepted"
    return "pending"


# ==================================================================================================


def merge_candidates(bench: dict, entries: List[dict], skipped: list) -> List[dict]:
    """New entries of an existing benchmark, without the papers it already has."""
    title = bench["title"]
    items_by_paper = collections.defaultdict(list)
    for item in bench["benchmark"]["items"]:
        p = item.get("p")
        if isinstance(p, dict):
            items_by_paper[utils.normalize_title(p.get("name"))].append(item)
            arxiv = utils.arxiv_id_of_link(p.get("link", ""))
            if arxiv:
                items_by_paper[arxiv].append(item)

    added = []
    for entry in entries:
        if not entry["mapped"]:
            skipped.append(skip_info(entry, "no mapped metric", title))
            continue
        keys = {
            utils.normalize_title(entry["p"]["name"]),
            utils.arxiv_id_of_link(entry["p"]["link"]),
            entry["evaluated_paper"],
        }
        existing = [it for k in keys if k for it in items_by_paper.get(k, [])]
        if existing:
            name = utils.normalize_title(entry["n"])
            if any(utils.normalize_title(it.get("n")) == name for it in existing):
                reason = "already in benchmark"
            elif any(
                utils.values_agree(v, it.get(k), min_digits=2) is not None
                for k, v in entry["mapped"].items()
                for it in existing
            ):
                reason = "same result already in benchmark"
            else:
                reason = "other variant of a paper in the benchmark"
            skipped.append(skip_info(entry, reason, title))
            continue
        added.append(entry)
    return added


def new_columns(bench: dict, added: List[dict], decisions: Dict[tuple, str]) -> tuple:
    """Approved new metric columns (set in the entries), and the undecided proposals."""
    fields = bench["benchmark"]["fields"]
    labels = {utils.normalize_title(f["label"]) for f in fields}
    counts = collections.Counter(m for e in added for m in e["extra"])
    next_key = 1 + max(
        int(f["key"][1:]) for f in fields if re.fullmatch(r"m\d+", f["key"])
    )
    approved, proposed = [], []
    for metric, count in counts.most_common():
        if count < max(3, len(added) / 2) or utils.normalize_title(metric) in labels:
            continue
        decision = decisions.get((bench["title"], metric))
        if decision is None:
            proposed.append(
                {
                    "benchmark": bench["title"],
                    "metric": metric,
                    "entries": count,
                    "existing_columns": [
                        f["label"] for f in fields if f["key"].startswith("m")
                    ],
                }
            )
        if decision != "add":
            continue
        key = f"m{next_key}"
        next_key += 1
        approved.append({"key": key, "label": metric[:64], "sortable": "true"})
        for e in added:
            if metric in e["extra"]:
                e["mapped"][key] = e["extra"][metric]
    return approved, proposed


def direction_of(metric: str, info: dict) -> bool:
    """True if higher values are better."""
    direction = (info.get("metrics") or {}).get(metric)
    if direction == "higher_is_better":
        return True
    if direction == "lower_is_better":
        return False
    return not LOWER_IS_BETTER_RE.search(metric)


def sort_items(items: List[dict], key: str, higher: bool) -> List[dict]:
    with_value = [it for it in items if utils.metric_value(it.get(key)) is not None]
    without = [it for it in items if utils.metric_value(it.get(key)) is None]
    with_value.sort(key=lambda it: it["d"] or "9999")
    with_value.sort(key=lambda it: utils.metric_value(it[key]), reverse=higher)
    without.sort(key=lambda it: it["d"] or "9999")
    return with_value + without


def dataset_info(dataset: dict, datasets: Dict[str, dict]) -> dict:
    parent = datasets.get(dataset.get("parent_dataset_id") or "", {})
    link = ""
    for ds in (dataset, parent):
        for url in (ds.get("homepage"), ds.get("url")):
            if valid_link(url) and "paperswithcode.co" not in url:
                link = link or url
        m = re.match(
            r"https://paperswithcode\.co/paper/(\d{4}\.\d{4,5})",
            ds.get("paper_url") or "",
        )
        if m:
            link = link or arxiv_abs(m.group(1))
        elif (
            valid_link(ds.get("paper_url"))
            and "paperswithcode.co" not in ds["paper_url"]
        ):
            link = link or ds["paper_url"]
    paper = clean_text(
        dataset.get("introducing_paper_title") or parent.get("introducing_paper_title")
    )
    return {"link": link, "paper": paper}


def build_new(entries: List[dict], info: dict) -> Tuple[Optional[dict], str]:
    """Fields and sorted entries of a new benchmark, or the reason it is too small."""
    counts = collections.Counter(m for e in entries for m in e["metrics"])
    if not counts:
        return None, "no metrics"
    primary = info.get("primary")
    if not primary or counts[primary] < len(entries) / 2:
        primary = counts.most_common(1)[0][0]
    metrics = [primary] + [
        m for m, c in counts.most_common() if m != primary and c >= len(entries) / 2
    ]
    metrics = metrics[:MAX_METRICS]
    keys = {m: f"m{i + 1}" for i, m in enumerate(metrics)}

    rows = []
    for e in entries:
        values = {keys[m]: v for m, v in e["metrics"].items() if m in keys}
        if values:
            e["values"] = values
            rows.append((item_of(e, values), e))

    with_primary = [it for it, _ in rows if "m1" in it]
    papers = {
        e["evaluated_paper"] if e["website"] else utils.normalize_title(it["p"]["name"])
        for it, e in rows
        if "m1" in it
    }
    if len(with_primary) < utils.MIN_ENTRIES or len(papers) < utils.MIN_PAPERS:
        return None, f"too small ({len(with_primary)} entries, {len(papers)} papers)"
    if not any(it["c"] for it, _ in rows):
        return None, "no code links"

    fields = (
        [FIELD_P, FIELD_C]
        + [{"key": keys[m], "label": m[:64], "sortable": "true"} for m in metrics]
        + [FIELD_N, FIELD_D]
    )
    order = {id(it): e for it, e in rows}
    items = sort_items([it for it, _ in rows], "m1", direction_of(primary, info))
    return {
        "fields": fields,
        "items": items,
        "entries": [order[id(it)] for it in items],
    }, ""


# ==================================================================================================


class Hierarchies:
    """Proposed task hierarchy of new benchmarks: the one of the benchmarks that other
    leaderboards of the task were merged into, a task of the same name, or the task
    chain of the site."""

    def __init__(self, benchmarks, matches, tasks):
        self.tasks = tasks
        self.by_task = collections.defaultdict(collections.Counter)
        for key, m in matches.items():
            if m["status"] == "merge":
                th = tuple(benchmarks[m["benchmark"]]["task-hierarchy"])
                self.by_task[key.split("/")[0]][th] += 1
        self.by_name = collections.defaultdict(collections.Counter)
        self.paths = collections.Counter()
        for data in benchmarks.values():
            th = data["task-hierarchy"]
            for i in range(len(th)):
                self.by_name[th[i].lower()][tuple(th[: i + 1])] += 1
                self.paths[" > ".join(th[: i + 1])] += 1

    def propose(self, task_id: str) -> List[str]:
        merged = self.by_task.get(task_id)
        if merged:
            th, count = merged.most_common(1)[0]
            if count >= 2 and count >= 0.6 * sum(merged.values()):
                return list(th)
        chain = utils.task_chain(self.tasks, task_id)
        for name in reversed(chain):
            paths = self.by_name.get(name.lower())
            if paths:
                return list(paths.most_common(1)[0][0])
        return chain


def sort_key(key: str) -> tuple:
    return tuple(int(x) for x in key.split("/"))


def leaderboard_info(keys: List[str]) -> dict:
    """Metric directions and primary metric of the first leaderboard that has them."""
    for key in keys:
        path = fetch_leaderboard_info.info_path(key)
        if os.path.isfile(path):
            return utils.load_json(path)
    return {}


def new_review(key: str, built: dict, hierarchies, tasks: dict, datasets: dict) -> dict:
    dataset = datasets[key.split("/")[1]]
    labels = {f["key"]: f["label"] for f in built["fields"]}
    return {
        "leaderboard": key,
        "task": " > ".join(utils.task_chain(tasks, key.split("/")[0])),
        "dataset": dataset.get("name"),
        "dataset_full_name": dataset.get("full_name"),
        "dataset_description": clean_text(
            dataset.get("short_description") or dataset.get("description")
        )[:400],
        "dataset_split": dataset.get("split_name"),
        "metrics": [f["label"] for f in built["fields"] if f["key"].startswith("m")],
        "proposed_task_hierarchy": hierarchies.propose(key.split("/")[0]),
        "num_entries": len(built["items"]),
        "entries": review_rows(built["entries"], labels),
    }


def benchmark_title(hierarchy: List[str], dataset: dict, taken: set) -> str:
    base = title = utils.benchmark_name(hierarchy[-1], dataset["name"])
    n = 2
    while title in taken:
        title = base[:124].strip("-") + f"-{n}"
        n += 1
    return title


def review_rows(entries: List[dict], metrics: Dict[str, str]) -> List[dict]:
    return [
        {
            "id": entry_id(e),
            "model": e["n"],
            "paper": e["p"]["name"][:70],
            "date": e["d"],
            **{metrics.get(k, k): v for k, v in e["values"].items()},
            "methodology": e["methodology"],
        }
        for e in entries[:MAX_REVIEW_ENTRIES]
    ]


def benchmark_summary(fields: List[dict], items: List[dict]) -> dict:
    metrics = {f["key"]: f["label"] for f in fields if f["key"].startswith("m")}
    sample = items[:5] + items[5:][-3:]
    return {
        "metrics": list(metrics.values()),
        "num_entries": len(items),
        "sample_entries": [
            {
                "model": it.get("n"),
                "paper": (
                    it["p"].get("name", "") if isinstance(it.get("p"), dict) else ""
                )[:70],
                "date": it.get("d"),
                **{metrics[k]: it[k] for k in metrics if it.get(k) not in (None, "")},
            }
            for it in sample
        ],
    }


# ==================================================================================================


def map_metrics(entries: List[dict], mapping: dict):
    """Values of mapped metrics in the benchmark's scale, and the other metrics."""
    for entry in entries:
        entry["mapped"] = {
            mapping[m]["key"]: utils.scale_value(v, mapping[m].get("factor", 1))
            for m, v in entry["metrics"].items()
            if m in mapping
        }
        entry["extra"] = {m: v for m, v in entry["metrics"].items() if m not in mapping}


def collect_entries(leaderboards, matches, known, details, skipped):
    """Entries of the leaderboards: by benchmark to merge into, by new leaderboard, and
    by title of the new benchmarks of split leaderboards."""
    merge_entries = collections.defaultdict(list)
    new_entries = {}
    split_new = {}
    for key, evals in sorted(leaderboards.items()):
        match = matches[key]
        entries = []
        for e in evals:
            entry, reason = make_entry(e, known, details)
            if reason:
                skipped.append(
                    {"reason": reason, "ids": [e["id"]], "model": e["model_name"]}
                )
                continue
            entries.append(entry)
        if match["status"] == "merge":
            map_metrics(entries, match["mapping"])
            merge_entries[match["benchmark"]].extend(entries)
        elif match["status"] == "new":
            new_entries[key] = entries
        for group in match.get("groups", []):
            part = [e for e in entries if e["ids"][0] in set(group["ids"])]
            if group["decision"] == "merge":
                map_metrics(part, group["metrics"])
                merge_entries[group["benchmark"]].extend(part)
                continue
            # optional column labels of the site's metrics, the primary one first
            labels = group.get("metrics")
            for e in part if labels else []:
                renamed = {}
                for m, v in e["metrics"].items():
                    if m in labels:
                        renamed.setdefault(labels[m], v)
                e["metrics"] = renamed
            th = group["task_hierarchy"]
            planned = split_new.setdefault(
                utils.benchmark_name(th[-1], group["dataset"]),
                {
                    "hierarchy": th,
                    "dataset_id": key.split("/")[1],
                    "labels": labels,
                    "keys": [],
                    "entries": [],
                },
            )
            planned["keys"].append(key)
            planned["entries"].extend(part)
    return merge_entries, new_entries, split_new


def plan_new_benchmarks(
    kept, split_new, *, new_entries, reviewed_new, benchmarks, datasets, skipped
):
    """
    Hierarchy, keys and entries of the new benchmarks by title: kept leaderboards of the
    same dataset as one, and the new groups of split leaderboards. An existing benchmark
    of the same dataset name and top-level task is a duplicate.
    """
    taken = set(benchmarks)
    existing = collections.defaultdict(list)
    for b, data in benchmarks.items():
        root = data["task-hierarchy"][0] if data["task-hierarchy"] else ""
        existing[(utils.compact_name(utils.bench_dataset_part(b)), root)].append(b)
    planned = {}
    for dataset_id, keys in kept.items():
        dataset = datasets[dataset_id]
        hierarchies_used = collections.Counter(
            tuple(reviewed_new[k]["task_hierarchy"]) for k in keys
        )
        hierarchy = list(hierarchies_used.most_common(1)[0][0])
        duplicate = existing.get(
            (utils.compact_name(utils.slugify(dataset["name"])), hierarchy[0])
        )
        if duplicate:
            skipped.append(
                {"reason": f"new benchmark may duplicate {duplicate}", "target": keys}
            )
            continue
        title = benchmark_title(hierarchy, dataset, taken)
        taken.add(title)
        planned[title] = {
            "hierarchy": hierarchy,
            "dataset_id": dataset_id,
            "keys": keys,
            "entries": [e for k in keys for e in new_entries[k]],
        }
    # parts of split leaderboards, their title names the protocol
    existing_titles = {utils.compact_name(b) for b in benchmarks}
    for title, plan in split_new.items():
        if utils.compact_name(title) in existing_titles:
            skipped.append(
                {"reason": f"new benchmark {title} exists", "target": plan["keys"]}
            )
        elif title in planned:
            planned[title]["keys"] += plan["keys"]
            planned[title]["entries"] += plan["entries"]
        else:
            planned[title] = plan
    return planned


def labelled_info(info: dict, labels: dict) -> dict:
    """Leaderboard info for the metrics renamed by a split review, the first is primary."""
    directions = info.get("metrics") or {}
    return {
        "primary": next(iter(labels.values())),
        "metrics": {label: directions.get(m) for m, label in reversed(labels.items())},
    }


def main():
    tasks, datasets, evaluations = utils.load_catalog()
    matches = utils.load_json(MATCHES_PATH)
    benchmarks = utils.load_benchmarks()
    overrides = utils.load_overrides()
    reviewed = utils.load_reviewed_evaluations()
    reviewed_new = {o["leaderboard"]: o for o in overrides.get("new_benchmarks", [])}
    column_decisions = {
        (o["benchmark"], o["metric"]): o["decision"]
        for o in overrides.get("new_columns", [])
    }

    leaderboards = collections.defaultdict(list)
    for e in evaluations:
        key = utils.leaderboard_key(e)
        if matches.get(key, {}).get("status") not in ("merge", "new", "split"):
            continue
        if utils.is_importable(e) and e["id"] not in reviewed["excluded"]:
            leaderboards[key].append(e)

    known = KnownPapers(benchmarks)
    details = load_paper_details(
        fetch_paper_details.needed_papers(evaluations, matches)
    )
    skipped = []
    merge_entries, new_entries, split_new = collect_entries(
        leaderboards, matches, known, details, skipped
    )

    if os.path.isdir(UPDATES_DIR):
        shutil.rmtree(UPDATES_DIR)
    entry_reviews, column_reviews = [], []
    stats = collections.Counter()

    # existing benchmarks
    for title, entries in sorted(merge_entries.items()):
        bench = benchmarks[title]
        added = merge_candidates(bench, combine(entries, skipped), skipped)
        fields, proposed = new_columns(bench, added, column_decisions)
        column_reviews.extend(proposed)
        for e in added:
            e["values"] = e["mapped"]
        accepted = [e for e in added if review_status(e, reviewed) == "accepted"]
        pending = [e for e in added if review_status(e, reviewed) == "pending"]
        if accepted:
            update = {
                "benchmark": title,
                "new_fields": fields,
                "entries": [item_of(e, e["mapped"]) for e in accepted],
            }
            utils.write_json(
                os.path.join(UPDATES_DIR, "merge", f"{title}.json"), update
            )
            stats["merged entries"] += len(accepted)
            stats["benchmarks with merged entries"] += 1
        if pending:
            all_fields = bench["benchmark"]["fields"] + fields
            labels = {f["key"]: f["label"] for f in all_fields}
            entry_reviews.append(
                {
                    "benchmark": title,
                    "task_hierarchy": bench["task-hierarchy"],
                    **benchmark_summary(all_fields, bench["benchmark"]["items"]),
                    "new_entries": review_rows(pending, labels),
                }
            )
            stats["entries to review"] += min(len(pending), MAX_REVIEW_ENTRIES)

    # new benchmarks: undecided ones are reviewed, kept ones of the same dataset
    # (listed under several tasks of the site) become one benchmark
    hierarchies = Hierarchies(benchmarks, matches, tasks)
    new_reviews = []
    kept = collections.defaultdict(list)
    for key, entries in sorted(new_entries.items(), key=lambda kv: sort_key(kv[0])):
        decision = reviewed_new.get(key, {}).get("decision")
        if decision == "keep":
            kept[key.split("/")[1]].append(key)
        if decision is not None:
            continue
        info = leaderboard_info([key])
        built, reason = (
            build_new(combine(entries, skipped), info) if info else (None, "")
        )
        if not built:
            reason = reason or "without leaderboard info"
            skipped.append({"reason": f"new benchmark {reason}", "target": key})
            continue
        new_reviews.append(new_review(key, built, hierarchies, tasks, datasets))

    planned = plan_new_benchmarks(
        kept,
        split_new,
        new_entries=new_entries,
        reviewed_new=reviewed_new,
        benchmarks=benchmarks,
        datasets=datasets,
        skipped=skipped,
    )
    for title, plan in planned.items():
        dataset, keys = datasets[plan["dataset_id"]], plan["keys"]
        info = leaderboard_info(keys)
        if plan.get("labels"):
            info = labelled_info(info, plan["labels"])
        built, reason = build_new(combine(plan["entries"], skipped), info)
        if not built:
            skipped.append({"reason": f"new benchmark {reason}", "target": keys})
            continue
        labels = {f["key"]: f["label"] for f in built["fields"]}
        accepted = [
            e for e in built["entries"] if review_status(e, reviewed) == "accepted"
        ]
        pending = [
            e for e in built["entries"] if review_status(e, reviewed) == "pending"
        ]
        final, reason = build_new(accepted, info)
        if pending:
            entry_reviews.append(
                {
                    "benchmark": title,
                    "task_hierarchy": plan["hierarchy"],
                    **benchmark_summary(
                        built["fields"], final["items"] if final else []
                    ),
                    "new_entries": review_rows(pending, labels),
                }
            )
            stats["entries to review"] += min(len(pending), MAX_REVIEW_ENTRIES)
        if not final:
            skipped.append(
                {"reason": f"new benchmark {reason} (reviewed)", "target": keys}
            )
            continue
        data = {
            "title": title,
            "task-hierarchy": plan["hierarchy"],
            "dataset-info": dataset_info(dataset, datasets),
            "benchmark": {"fields": final["fields"], "items": final["items"]},
        }
        utils.write_json(os.path.join(UPDATES_DIR, "new", f"{title}.json"), data)
        stats["new benchmarks"] += 1
        stats["new benchmark entries"] += len(final["items"])

    utils.write_json(os.path.join(UPDATES_DIR, "skipped.json"), skipped, compact=True)
    os.makedirs(utils.REVIEW_DIR, exist_ok=True)
    n_entry_files = utils.write_review_batches(
        "entries", entry_reviews, "new_entries", REVIEW_BATCH_ENTRIES
    )
    n_new_files = utils.write_review_batches(
        "new", new_reviews, "entries", REVIEW_BATCH_ENTRIES
    )
    utils.write_json(os.path.join(utils.REVIEW_DIR, "new_columns.json"), column_reviews)
    utils.write_json(
        os.path.join(utils.REVIEW_DIR, "task_hierarchies.json"),
        sorted(hierarchies.paths),
        compact=True,
    )

    print("Updates:", dict(stats))
    print(
        f"Reviews: {len(entry_reviews)} benchmarks with entries in {n_entry_files} files, "
        f"{len(new_reviews)} new benchmarks in {n_new_files} files, "
        f"{len(column_reviews)} new columns"
    )
    print(
        "Skipped:",
        dict(collections.Counter(s["reason"] for s in skipped).most_common()),
    )


# ==================================================================================================

if __name__ == "__main__":
    main()
