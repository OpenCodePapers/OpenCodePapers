"""
Find the existing benchmark of each leaderboard (task + dataset). Writes data/matches.json
and the leaderboards to review (data/review/matches_*.json). Decisions in data/overrides.json
take precedence.

Leaderboards are matched by the results both list for the same papers (title or arXiv id):
- Values agree if they are equal after rounding to the lesser precision (also x100 or
  /100) and have 3+ significant digits. A metric pair needs agreeing values in 2+ papers
  and in half of the papers that report both. Metrics with the same label are mapped too.
- A benchmark is accepted with agreeing values in 3+ papers and at most 25% disagreeing
  ones (5+ and 20% if the dataset names are not similar).
- Leaderboards with weaker evidence, or a benchmark of a similar dataset name, are
  reviewed. The others become new benchmarks, or are dropped if they are too small.
- Leaderboards dropped by a review (often mixing protocols, like zero-shot and fine-tuned
  results) are reviewed again entry by entry (data/review/split_*.json): their entries are
  split into groups of one protocol, which are merged or become new benchmarks.
"""

import collections
import os
import re
import statistics
from typing import Dict, List, Set

import utils

# ==================================================================================================

MATCHES_PATH = "data/matches.json"
REVIEW_BATCH = 20
SPLIT_BATCH_ENTRIES = 250
MAX_SPLIT_CANDIDATES = 6
# dataset links used by more benchmarks don't identify a dataset (like cocodataset.org)
MAX_LINK_BENCHMARKS = 10

MIN_AGREEING_PAPERS = 3
MAX_CONFLICT_SHARE = 0.25
MIN_PAIR_PAPERS = 2
MIN_SHARED_PAPERS = 2
# for benchmarks of a dissimilar dataset name
STRONG_AGREEING_PAPERS = 5
STRONG_CONFLICT_SHARE = 0.2

# Words that don't identify a dataset in its name
GENERIC_TOKENS = frozenset(
    "test val validation dev set dataset datasets benchmark bench the of and a for "
    "on in with data split full official".split()
)

# ==================================================================================================


def dataset_tokens(text: str) -> Set[str]:
    return set(utils.normalize_title(text).split()) - GENERIC_TOKENS


def normalize_link(link: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", (link or "").strip().lower()).rstrip("/")


class BenchIndex:
    """Benchmark items by paper, and benchmarks by paper and dataset name."""

    def __init__(self, benchmarks: Dict[str, dict]):
        self.benchmarks = benchmarks
        self.items = {}
        self.by_paper = collections.defaultdict(set)
        self.by_token = collections.defaultdict(set)
        self.by_compact = collections.defaultdict(set)
        self.by_link = collections.defaultdict(set)
        for title, data in benchmarks.items():
            by_title = collections.defaultdict(list)
            by_arxiv = collections.defaultdict(list)
            for item in data["benchmark"]["items"]:
                p = item.get("p")
                if not isinstance(p, dict):
                    continue
                t = utils.normalize_title(p.get("name"))
                a = utils.arxiv_id_of_link(p.get("link", ""))
                by_title[t].append(item)
                self.by_paper[("t", t)].add(title)
                if a:
                    by_arxiv[a].append(item)
                    self.by_paper[("a", a)].add(title)
            self.items[title] = (by_title, by_arxiv)
            for tok in dataset_tokens(
                utils.bench_dataset_part(title).replace("-", " ")
            ):
                self.by_token[tok].add(title)
            self.by_compact[utils.compact_name(utils.bench_dataset_part(title))].add(
                title
            )
            link = normalize_link(data["dataset-info"].get("link"))
            if link:
                self.by_link[link].add(title)

    def paper_items(self, bench: str, paper: tuple) -> List[dict]:
        by_title, by_arxiv = self.items[bench]
        return (
            by_title.get(paper[0]) or (by_arxiv.get(paper[1]) if paper[1] else []) or []
        )

    def paper_benches(self, paper: tuple) -> Set[str]:
        found = set(self.by_paper.get(("t", paper[0]), set()))
        if paper[1]:
            found |= self.by_paper.get(("a", paper[1]), set())
        return found

    def variant_candidates(self, dataset: dict) -> Dict[str, float]:
        """
        Benchmarks with the most distinctive word of the dataset name (like "calvin" of
        "CALVIN ABCD to D"), and the share of words both names have.
        """
        toks = dataset_tokens(dataset["name"].replace("-", " "))
        known = [
            t for t in toks if len(t) >= 3 and not t.isdigit() and self.by_token.get(t)
        ]
        if not known:
            return {}
        rare = min(known, key=lambda t: (len(self.by_token[t]), t))
        found = {}
        for bench in self.by_token[rare]:
            btoks = dataset_tokens(utils.bench_dataset_part(bench).replace("-", " "))
            found[bench] = len(toks & btoks) / len(toks | btoks)
        return found

    def link_candidates(self, dataset: dict) -> Set[str]:
        """Benchmarks with the dataset's homepage or paper as link (like InfographicVQA
        for "InfoVQA")."""
        links = {normalize_link(dataset.get(k)) for k in ("homepage", "url")}
        m = re.search(r"/paper/(\d{4}\.\d{4,5})", dataset.get("paper_url") or "")
        if m:
            links.add(f"arxiv.org/abs/{m.group(1)}")
        found = set()
        for link in links - {""}:
            benches = self.by_link.get(link, set())
            if len(benches) <= MAX_LINK_BENCHMARKS:
                found |= benches
        return found

    def name_candidates(self, dataset: dict) -> Dict[str, bool]:
        """
        Benchmarks whose dataset part of the title is similar to the dataset name,
        and whether it is the same name. Distinctive words of the name (used by
        few benchmarks, like "seaquest" in "Atari 2600 Seaquest") must all be in it.
        """
        found = {}
        compact = utils.compact_name(dataset["name"])
        for bench in self.by_compact.get(compact, ()):
            found[bench] = True
        for name in (dataset["name"], dataset.get("slug") or ""):
            toks = dataset_tokens(name.replace("-", " "))
            if not toks:
                continue
            exact = utils.normalize_title(name)
            pool = set.union(*(self.by_token.get(t, set()) for t in toks))
            for bench in pool:
                part = utils.bench_dataset_part(bench).replace("-", " ")
                btoks = dataset_tokens(part)
                inter = toks & btoks
                rare = {t for t in toks if len(self.by_token.get(t, ())) <= 20}
                if not rare <= btoks:
                    continue
                if not any(len(t) >= 3 and not t.isdigit() for t in inter):
                    continue
                if len(inter) / len(toks | btoks) >= 0.5:
                    found[bench] = found.get(bench, False) or part == exact
        return found


# ==================================================================================================


def paper_key(evaluation: dict) -> tuple:
    arxiv = (evaluation.get("paper_arxiv_id") or "").split("v")[0]
    return (utils.normalize_title(evaluation.get("paper_title")), arxiv)


def metric_fields(data: dict) -> Dict[str, str]:
    return {
        f["key"]: f["label"]
        for f in data["benchmark"]["fields"]
        if f["key"].startswith("m")
    }


def guess_factor(hf_values: List[float], ocp_values: List[float]) -> int:
    if not hf_values or not ocp_values:
        return 1
    a, b = statistics.median(hf_values), statistics.median(ocp_values)
    if a == 0 or b == 0:
        return 1
    ratio = abs(b / a)
    if 30 < ratio < 300:
        return 100
    if 1 / 300 < ratio < 1 / 30:
        return -100
    return 1


def score_candidate(
    index: BenchIndex, bench: str, papers: Dict[tuple, List[dict]]
) -> dict:
    """Metric mapping and evidence of a leaderboard (evaluations by paper) for a benchmark."""
    data = index.benchmarks[bench]
    fields = metric_fields(data)
    agree = collections.defaultdict(set)
    comparable = collections.defaultdict(set)
    factors = collections.defaultdict(collections.Counter)

    shared = 0
    examples = []
    for paper, evs in papers.items():
        items = index.paper_items(bench, paper)
        if not items:
            continue
        shared += 1
        if len(examples) < 4:
            examples.append(
                {
                    "paper": paper[0][:80],
                    "leaderboard": [
                        {"model": e["model_name"], **e["metrics"]} for e in evs[:3]
                    ],
                    "benchmark": [
                        {
                            "model": it.get("n"),
                            **{k: it[k] for k in fields if it.get(k)},
                        }
                        for it in items[:3]
                    ],
                }
            )
        hf_metrics = {m for e in evs for m in e["metrics"]}
        for hm in hf_metrics:
            hvals = [e["metrics"][hm] for e in evs if hm in e["metrics"]]
            for ok in fields:
                ovals = [it[ok] for it in items if it.get(ok) not in (None, "")]
                if not ovals:
                    continue
                comparable[(hm, ok)].add(paper)
                for hv in hvals:
                    for ov in ovals:
                        f = utils.values_agree(hv, ov)
                        if f:
                            agree[(hm, ok)].add(paper)
                            factors[(hm, ok)][f] += 1

    # value based mapping, one to one, best supported pairs first
    mapping = {}
    pairs = sorted(agree, key=lambda k: (-len(agree[k]), k))
    for hm, ok in pairs:
        n_agree, n_comp = len(agree[(hm, ok)]), len(comparable[(hm, ok)])
        if hm in mapping or ok in {m["key"] for m in mapping.values()}:
            continue
        if n_agree >= MIN_PAIR_PAPERS and n_agree / n_comp >= 0.5:
            mapping[hm] = {
                "key": ok,
                "factor": factors[(hm, ok)].most_common(1)[0][0],
                "papers": n_agree,
                "by": "values",
            }

    agree_papers = set().union(*(agree[(hm, m["key"])] for hm, m in mapping.items()))
    comp_papers = set().union(
        *(comparable[(hm, m["key"])] for hm, m in mapping.items())
    )
    conflicts = comp_papers - agree_papers
    best_any = max((len(v) for v in agree.values()), default=0)
    return {
        "benchmark": bench,
        "shared_papers": shared,
        "agreeing_papers": len(agree_papers),
        "conflicting_papers": len(conflicts),
        "best_single_pair": best_any,
        "mapping": mapping,
        "shared_examples": examples,
        "_comparable": comparable,
        "_agree": agree,
    }


def add_label_mappings(cand: dict, data: dict, evals: List[dict]):
    """Map remaining hf metrics to unmapped benchmark metrics with the same label."""
    fields = metric_fields(data)
    used = {m["key"] for m in cand["mapping"].values()}
    labels = {
        utils.normalize_title(lbl): k for k, lbl in fields.items() if k not in used
    }
    hf_metrics = collections.Counter(m for e in evals for m in e["metrics"])
    for hm in hf_metrics:
        ok = labels.get(utils.normalize_title(hm))
        if hm in cand["mapping"] or not ok or ok in used:
            continue
        n_comp = len(cand["_comparable"].get((hm, ok), ()))
        n_agree = len(cand["_agree"].get((hm, ok), ()))
        if n_comp and n_agree / n_comp < 0.5:
            continue
        hvals = [
            utils.metric_value(e["metrics"][hm]) for e in evals if hm in e["metrics"]
        ]
        ovals = [utils.metric_value(it.get(ok)) for it in data["benchmark"]["items"]]
        hvals = [v for v in hvals if v is not None]
        ovals = [v for v in ovals if v is not None]
        cand["mapping"][hm] = {
            "key": ok,
            "factor": guess_factor(hvals, ovals),
            "papers": n_agree,
            "by": "label",
        }
        used.add(ok)


def accepted(cand: dict, strong: bool = False) -> bool:
    n_agree, n_conf = cand["agreeing_papers"], cand["conflicting_papers"]
    min_agree, max_share = MIN_AGREEING_PAPERS, MAX_CONFLICT_SHARE
    if strong:
        min_agree, max_share = STRONG_AGREEING_PAPERS, STRONG_CONFLICT_SHARE
    if n_agree < min_agree:
        return False
    return n_conf <= max_share * (n_agree + n_conf)


# ==================================================================================================


def review_info(key, evals, cands, index, *, tasks, datasets) -> dict:
    """Compact description of a leaderboard and its candidate benchmarks for reviewers."""
    task_id, dataset_id = key.split("/")
    ds = datasets.get(dataset_id, {})
    metrics = collections.Counter(m for e in evals for m in e["metrics"])
    samples = sorted(evals, key=lambda e: e["paper_published_date"] or "")
    samples = samples[:3] + samples[3:][-7:]
    info = {
        "leaderboard": key,
        "task": " > ".join(utils.task_chain(tasks, task_id)),
        "dataset": ds.get("name"),
        "dataset_full_name": ds.get("full_name"),
        "dataset_description": (
            ds.get("short_description") or ds.get("description") or ""
        )[:400],
        "dataset_split": ds.get("split_name"),
        "dataset_paper": ds.get("introducing_paper_title"),
        "metrics": dict(metrics.most_common()),
        "num_entries": len(evals),
        "num_papers": len({paper_key(e) for e in evals}),
        "sample_entries": [
            {
                "model": e["model_name"],
                "paper": (e["paper_title"] or "")[:80],
                "date": e["paper_published_date"],
                "metrics": e["metrics"],
                "methodology": (e.get("methodology") or "")[:200],
            }
            for e in samples
        ],
        "candidates": [candidate_info(c, index) for c in cands[:4]],
    }
    return info


def candidate_info(cand: dict, index: BenchIndex) -> dict:
    data = index.benchmarks[cand["benchmark"]]
    fields = metric_fields(data)
    items = data["benchmark"]["items"]
    items = items[:3] + items[-2:] if len(items) > 5 else items
    return {
        "benchmark": cand["benchmark"],
        "task_hierarchy": data["task-hierarchy"],
        "dataset_info": data["dataset-info"],
        "metrics": fields,
        "num_entries": len(data["benchmark"]["items"]),
        "shared_papers": cand["shared_papers"],
        "agreeing_papers": cand["agreeing_papers"],
        "conflicting_papers": cand["conflicting_papers"],
        "value_mapping": {hm: m["key"] for hm, m in cand["mapping"].items()},
        "shared_paper_examples": cand["shared_examples"],
        "sample_entries": [
            {
                "model": it.get("n"),
                "paper": (it["p"].get("name", "") if isinstance(it["p"], dict) else "")[
                    :80
                ],
                "date": it.get("d"),
                "metrics": {k: it[k] for k in fields if it.get(k) not in (None, "")},
            }
            for it in items
        ],
    }


def split_candidates(cands: List[dict], variants: Dict[str, float]) -> List[dict]:
    """Benchmarks of a similar dataset name or with agreeing values first, then other
    benchmarks of the dataset (protocol variants)."""
    similar = [
        c
        for c in cands
        if variants.get(c["benchmark"], 0) >= 0.5
        or c["agreeing_papers"] >= MIN_AGREEING_PAPERS
    ]
    similar.sort(
        key=lambda c: (-c["agreeing_papers"], -variants.get(c["benchmark"], 0))
    )
    others = [c for c in cands if c not in similar and c["benchmark"] in variants]
    others.sort(key=lambda c: (-variants[c["benchmark"]], -c["agreeing_papers"]))
    return (similar + others)[:MAX_SPLIT_CANDIDATES]


def split_review_info(key, evals, cands, index, *, tasks, datasets, reason) -> dict:
    """All entries of a dropped leaderboard, with the candidates that list their papers
    with the same or another value."""
    task_id, dataset_id = key.split("/")
    ds = datasets.get(dataset_id, {})
    rows = []
    for e in sorted(evals, key=lambda e: (e["paper_published_date"] or "", e["id"])):
        same, other = [], []
        for cand in cands:
            data = index.benchmarks[cand["benchmark"]]
            items = index.paper_items(cand["benchmark"], paper_key(e))
            if not items:
                continue
            agree = any(
                utils.values_agree(v, it.get(k)) is not None
                for v in e["metrics"].values()
                for it in items
                for k in metric_fields(data)
            )
            (same if agree else other).append(cand["benchmark"])
        leaderboard = utils.leaderboard_source(e)
        reporting = utils.reporting_paper_ref(e)
        row = {
            "id": e["id"],
            "model": e["model_name"],
            "paper": (e["paper_title"] or "")[:80],
            "source": (
                f"website {leaderboard[0]}"
                if leaderboard
                else f"reported by paper {reporting}" if reporting else ""
            ),
            "date": e["paper_published_date"],
            "metrics": e["metrics"],
            "methodology": (e.get("methodology") or "")[:200],
            "same_value_in": same,
            "other_value_in": other,
        }
        rows.append({k: v for k, v in row.items() if v})
    return {
        "leaderboard": key,
        "task": " > ".join(utils.task_chain(tasks, task_id)),
        "dataset": ds.get("name"),
        "dataset_full_name": ds.get("full_name"),
        "dataset_description": (
            ds.get("short_description") or ds.get("description") or ""
        )[:400],
        "dataset_split": ds.get("split_name"),
        "drop_reason": reason,
        "candidates": [
            {
                k: v
                for k, v in candidate_info(c, index).items()
                if k != "shared_paper_examples"
            }
            for c in cands
        ],
        "entries": rows,
    }


def apply_override(entry: dict, ov: dict, index: BenchIndex):
    decision = ov.get("decision")
    if decision == "merge":
        bench = ov["benchmark"]
        if bench not in index.benchmarks:
            print(f"Override for {entry['leaderboard']}: unknown benchmark {bench}")
            return
        entry["status"] = "merge"
        entry["benchmark"] = bench
        entry["mapping"] = {
            hm: {"key": m["key"], "factor": m.get("factor", 1), "by": "review"}
            for hm, m in ov["metrics"].items()
        }
    elif decision in ("new", "drop"):
        entry["status"] = decision
        entry.pop("benchmark", None)
        entry.pop("mapping", None)
    entry["reviewed"] = ov.get("reason", True)


# ==================================================================================================


def main():
    tasks, datasets, evaluations = utils.load_catalog()
    index = BenchIndex(utils.load_benchmarks())
    all_overrides = utils.load_overrides()
    overrides = {o["leaderboard"]: o for o in all_overrides.get("leaderboards", [])}
    splits = {o["leaderboard"]: o for o in all_overrides.get("splits", [])}
    dropped_new = {
        o["leaderboard"]: o.get("reason", "")
        for o in all_overrides.get("new_benchmarks", [])
        if o["decision"] == "drop"
    }

    leaderboards = collections.defaultdict(list)
    for e in evaluations:
        if utils.is_importable(e):
            leaderboards[utils.leaderboard_key(e)].append(e)

    matches = {}
    review, split_review = [], []
    stats = collections.Counter()
    for key in sorted(leaderboards, key=lambda k: tuple(int(x) for x in k.split("/"))):
        evals = leaderboards[key]
        papers = collections.defaultdict(list)
        for e in evals:
            papers[paper_key(e)].append(e)

        bench_counts = collections.Counter()
        for paper in papers:
            for bench in index.paper_benches(paper):
                bench_counts[bench] += 1
        dataset = datasets.get(key.split("/")[1], {"name": ""})
        names = index.name_candidates(dataset)
        cand_benches = {b for b, c in bench_counts.items() if c >= MIN_SHARED_PAPERS}
        cand_benches |= set(names)

        cands = [score_candidate(index, b, papers) for b in sorted(cand_benches)]
        cands.sort(
            key=lambda c: (
                -c["agreeing_papers"],
                c["conflicting_papers"],
                c["benchmark"],
            )
        )
        good = [c for c in cands if accepted(c, strong=c["benchmark"] not in names)]

        entry = {
            "leaderboard": key,
            "task": tasks.get(key.split("/")[0], {}).get("name"),
            "dataset": dataset["name"],
            "num_entries": len(evals),
            "num_papers": len(papers),
        }
        if good:
            best = good[0]
            add_label_mappings(best, index.benchmarks[best["benchmark"]], evals)
            entry.update(
                status="merge",
                benchmark=best["benchmark"],
                mapping=best["mapping"],
                evidence={
                    k: best[k]
                    for k in ("shared_papers", "agreeing_papers", "conflicting_papers")
                },
                also_accepted=[c["benchmark"] for c in good[1:]],
            )
        else:
            # some evidence for an existing benchmark: review, else new benchmark
            # (small leaderboards without agreeing values need an equal dataset name)
            big = len(evals) >= utils.MIN_ENTRIES and len(papers) >= utils.MIN_PAPERS
            hints = [
                c
                for c in cands
                if c["best_single_pair"] >= 1
                or accepted(c)
                or (c["benchmark"] in names and (big or names[c["benchmark"]]))
            ]
            if hints:
                entry["status"] = "review"
                entry["candidates"] = [c["benchmark"] for c in hints[:5]]
                review.append(
                    review_info(
                        key, evals, hints, index, tasks=tasks, datasets=datasets
                    )
                )
            else:
                entry["status"] = "new" if big else "drop"

        if key in overrides:
            apply_override(entry, overrides[key], index)

        if key in splits:
            for k in (
                "benchmark",
                "mapping",
                "candidates",
                "evidence",
                "also_accepted",
            ):
                entry.pop(k, None)
            entry.update(status="split", groups=splits[key]["groups"])
        elif len(evals) >= utils.MIN_ENTRIES and (
            (entry["status"] == "drop" and entry.get("reviewed")) or key in dropped_new
        ):
            variants = index.variant_candidates(dataset)
            for bench in index.link_candidates(dataset):
                variants[bench] = max(variants.get(bench, 0), 0.5)
            variants.update({b: 1.0 for b in names})
            cands += [
                score_candidate(index, b, papers)
                for b in sorted(set(variants) - cand_benches)
            ]
            split_review.append(
                split_review_info(
                    key,
                    evals,
                    split_candidates(cands, variants),
                    index,
                    tasks=tasks,
                    datasets=datasets,
                    reason=dropped_new.get(key) or entry.get("reviewed"),
                )
            )
        stats[entry["status"]] += 1
        matches[key] = entry

    utils.write_json(MATCHES_PATH, matches)

    # reviews still missing a decision, in batches
    os.makedirs(utils.REVIEW_DIR, exist_ok=True)
    for fname in os.listdir(utils.REVIEW_DIR):
        if re.fullmatch(r"matches_\d+\.json", fname):
            os.remove(os.path.join(utils.REVIEW_DIR, fname))
    todo = [
        r for r in review if r["leaderboard"] not in overrides.keys() | splits.keys()
    ]
    for i in range(0, len(todo), REVIEW_BATCH):
        path = os.path.join(utils.REVIEW_DIR, f"matches_{i // REVIEW_BATCH:03d}.json")
        utils.write_review(path, todo[i : i + REVIEW_BATCH])
    n_split_files = utils.write_review_batches(
        "split", split_review, "entries", SPLIT_BATCH_ENTRIES
    )

    merged_into = collections.Counter(
        m["benchmark"] for m in matches.values() if m.get("benchmark")
    )
    print(f"Leaderboards: {len(matches)} {dict(stats)}")
    print(f"Benchmarks receiving merges: {len(merged_into)}")
    print(
        f"Reviews pending: {len(todo)} in {(len(todo) + REVIEW_BATCH - 1) // REVIEW_BATCH} files, "
        f"{len(split_review)} leaderboards to split in {n_split_files} files"
    )


# ==================================================================================================

if __name__ == "__main__":
    main()
