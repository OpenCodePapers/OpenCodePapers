"""
Add the review decisions (data/review/*.decisions.json) to data/overrides.json
and data/reviewed_evaluations.json.

Invalid decisions are reported and skipped, so they are reviewed again. The entries
shown in a review are accepted if their benchmark is kept and they aren't excluded, the
ones of split leaderboards if they are in a group. With --check <file> a split decisions
file is only checked (for reviewers).
"""

import glob
import os
import sys

import utils

# ==================================================================================================

REVIEW_DIR = utils.REVIEW_DIR
APPLIED_DIR = "data/review/applied/"
MATCHES_PATH = "data/matches.json"

# ==================================================================================================


def check_match(dec: dict, matches: dict, benchmarks: dict) -> str:
    """Error message of an invalid leaderboard decision, else ""."""
    key = dec.get("leaderboard")
    if key not in matches:
        return f"unknown leaderboard {key}"
    decision = dec.get("decision")
    if decision not in ("merge", "new", "drop"):
        return f"{key}: unknown decision {decision}"
    if decision == "new":
        m = matches[key]
        if m["num_entries"] < utils.MIN_ENTRIES or m["num_papers"] < utils.MIN_PAPERS:
            return f"{key}: too small for a new benchmark"
    if decision == "merge":
        return check_mapping(key, dec, benchmarks)
    return ""


def check_mapping(key: str, dec: dict, benchmarks: dict) -> str:
    bench = dec.get("benchmark")
    if bench not in benchmarks:
        return f"{key}: unknown benchmark {bench}"
    keys = {f["key"] for f in benchmarks[bench]["benchmark"]["fields"]}
    metrics = dec.get("metrics") or {}
    if not metrics:
        return f"{key}: merge without metric mapping"
    for hm, m in metrics.items():
        if m.get("key") not in keys or not m["key"].startswith("m"):
            return f"{key}: unknown metric key {m.get('key')} for {hm}"
        if m.get("factor", 1) not in (1, 100, -100):
            return f"{key}: invalid factor {m.get('factor')} for {hm}"
    return ""


def check_hierarchy(key: str, th) -> str:
    if (
        not isinstance(th, list)
        or not th
        or not all(isinstance(t, str) and t for t in th)
    ):
        return f"{key}: invalid task hierarchy {th}"
    if any(len(t) > 96 for t in th):
        return f"{key}: task hierarchy item longer than 96 characters"
    return ""


def check_new(dec: dict, shown: dict) -> str:
    key = dec.get("leaderboard")
    if key not in shown:
        return f"leaderboard {key} not in the review file"
    if dec.get("decision") not in ("keep", "drop"):
        return f"{key}: unknown decision {dec.get('decision')}"
    if dec["decision"] == "keep":
        return check_hierarchy(key, dec.get("task_hierarchy"))
    return ""


def check_split(dec: dict, shown: dict, benchmarks: dict) -> str:
    """Groups of entries (ids of the review file), merged or new benchmarks."""
    key = dec.get("leaderboard")
    if key not in shown:
        return f"leaderboard {key} not in the review file"
    if not isinstance(dec.get("groups"), list):
        return f"{key}: no groups"
    seen = set()
    for group in dec["groups"]:
        ids = {str(i) for i in group.get("ids") or []}
        if not ids or not ids <= shown[key]:
            return f"{key}: group without ids or with unknown ids"
        if ids & seen:
            return f"{key}: ids in several groups"
        seen |= ids
        if group.get("decision") == "merge":
            error = check_mapping(key, group, benchmarks)
        elif group.get("decision") == "new":
            error = check_hierarchy(key, group.get("task_hierarchy"))
            if not error and not isinstance(group.get("dataset"), str):
                error = f"{key}: new group without dataset name"
            labels = group.get("metrics", {})
            if not error and (
                not isinstance(labels, dict)
                or not all(
                    isinstance(v, str) and 0 < len(v) <= 64 for v in labels.values()
                )
            ):
                error = f"{key}: invalid column labels {labels}"
            if not error:
                title = utils.benchmark_name(
                    group["task_hierarchy"][-1], group["dataset"]
                )
                if title in benchmarks:
                    error = f"{key}: {title} exists, merge instead"
        else:
            error = f"{key}: unknown group decision {group.get('decision')}"
        if error:
            return error
    return ""


def check_entries(dec: dict, shown: dict) -> str:
    bench = dec.get("benchmark")
    if bench not in shown:
        return f"benchmark {bench} not in the review file"
    if dec.get("decision") not in ("keep", "drop"):
        return f"{bench}: unknown decision {dec.get('decision')}"
    return ""


# ==================================================================================================


def archive(path: str):
    os.makedirs(APPLIED_DIR, exist_ok=True)
    name = os.path.basename(path)
    n = 0
    while os.path.exists(os.path.join(APPLIED_DIR, f"{n:03d}_{name}")):
        n += 1
    os.rename(path, os.path.join(APPLIED_DIR, f"{n:03d}_{name}"))


def load_decisions(path: str):
    try:
        return utils.load_json(path)
    except ValueError as e:
        print(f"{path}: invalid JSON: {e}")
        return None


def apply_matches(overrides: dict, matches: dict, benchmarks: dict):
    entries = {o["leaderboard"]: o for o in overrides.get("leaderboards", [])}
    n = 0
    for path in sorted(glob.glob(os.path.join(REVIEW_DIR, "matches_*.decisions.json"))):
        decisions = load_decisions(path)
        if decisions is None:
            continue
        archive(path)
        for dec in decisions:
            error = check_match(dec, matches, benchmarks)
            if error:
                print(f"{os.path.basename(path)}: {error}")
                continue
            m = matches[dec["leaderboard"]]
            entries[dec["leaderboard"]] = {
                "leaderboard": dec["leaderboard"],
                "task": m["task"],
                "dataset": m["dataset"],
                **{k: v for k, v in dec.items() if k != "leaderboard"},
            }
            n += 1
    overrides["leaderboards"] = sorted(entries.values(), key=sort_key)
    print(f"leaderboards: {n} reviewed, {len(entries)} in overrides")


def apply_entry_reviews(
    prefix: str, key_name: str, check, overrides: dict, reviewed: dict
):
    """Decisions of new_* or entries_* reviews; their shown entries are accepted or excluded."""
    new_benchmarks = {o["leaderboard"]: o for o in overrides.get("new_benchmarks", [])}
    n_decisions, n_ids = 0, 0
    for path in sorted(
        glob.glob(os.path.join(REVIEW_DIR, f"{prefix}_*.decisions.json"))
    ):
        input_path = path.replace(".decisions.json", ".json")
        if not os.path.isfile(input_path):
            print(f"{path}: review file {input_path} not found")
            continue
        entries_key = "entries" if prefix == "new" else "new_entries"
        shown = {
            g[key_name]: [i for row in g[entries_key] for i in row["id"].split("+")]
            for g in utils.load_json(input_path)
        }
        decisions = load_decisions(path)
        if decisions is None:
            continue
        archive(path)
        archive(input_path)
        for dec in decisions:
            error = check(dec, shown)
            if error:
                print(f"{os.path.basename(path)}: {error}")
                continue
            excluded = {
                i for row_id in dec.get("exclude", []) for i in str(row_id).split("+")
            }
            ids = shown[dec[key_name]]
            if dec["decision"] == "keep":
                reviewed["accepted"].update(i for i in ids if i not in excluded)
                reviewed["excluded"].update(i for i in ids if i in excluded)
            else:
                reviewed["excluded"].update(ids)
            if prefix == "new":
                new_benchmarks[dec["leaderboard"]] = {
                    k: dec[k]
                    for k in ("leaderboard", "decision", "task_hierarchy", "reason")
                    if k in dec
                }
            n_decisions += 1
            n_ids += len(ids)
    if prefix == "new":
        overrides["new_benchmarks"] = sorted(new_benchmarks.values(), key=sort_key)
    print(f"{prefix}: {n_decisions} reviewed with {n_ids} entries")


def apply_splits(overrides: dict, matches: dict, benchmarks: dict, reviewed: dict):
    """Entries of split leaderboards are accepted if they are in a group."""
    splits = {o["leaderboard"]: o for o in overrides.get("splits", [])}
    n_decisions, n_accepted, n_excluded = 0, 0, 0
    for path in sorted(glob.glob(os.path.join(REVIEW_DIR, "split_*.decisions.json"))):
        input_path = path.replace(".decisions.json", ".json")
        if not os.path.isfile(input_path):
            print(f"{path}: review file {input_path} not found")
            continue
        shown = split_shown(utils.load_json(input_path))
        decisions = load_decisions(path)
        if decisions is None:
            continue
        archive(path)
        archive(input_path)
        for dec in decisions:
            error = check_split(dec, shown, benchmarks)
            if error:
                print(f"{os.path.basename(path)}: {error}")
                continue
            key = dec["leaderboard"]
            groups = [
                {**g, "ids": sorted({str(i) for i in g["ids"]}, key=int)}
                for g in dec["groups"]
            ]
            accepted = {i for g in groups for i in g["ids"]}
            reviewed["excluded"] -= accepted
            reviewed["excluded"] |= shown[key] - accepted
            reviewed["accepted"] |= accepted
            splits[key] = {
                "leaderboard": key,
                "task": matches[key]["task"],
                "dataset": matches[key]["dataset"],
                "groups": groups,
                "reason": dec.get("reason", ""),
            }
            n_decisions += 1
            n_accepted += len(accepted)
            n_excluded += len(shown[key] - accepted)
    overrides["splits"] = sorted(splits.values(), key=sort_key)
    print(
        f"split: {n_decisions} reviewed, {n_accepted} entries in groups, "
        f"{n_excluded} excluded"
    )


def split_shown(leaderboards: list) -> dict:
    return {
        lb["leaderboard"]: {row["id"] for row in lb["entries"]} for lb in leaderboards
    }


def check_split_file(path: str, benchmarks: dict):
    shown = split_shown(utils.load_json(path.replace(".decisions.json", ".json")))
    decisions = load_decisions(path) or []
    errors = [check_split(dec, shown, benchmarks) for dec in decisions]
    if [dec.get("leaderboard") for dec in decisions] != list(shown):
        errors.append("leaderboards missing or not in the order of the review file")
    for error in filter(None, errors):
        print(error)
    n_ids = sum(
        len(g.get("ids", [])) for dec in decisions for g in dec.get("groups", [])
    )
    print(
        f"{len(decisions)} decisions, {n_ids} entries in groups, "
        f"{len(list(filter(None, errors)))} errors"
    )


def sort_key(entry: dict):
    return tuple(int(x) for x in entry["leaderboard"].split("/"))


# ==================================================================================================


def main():
    args = sys.argv[1:]
    if args:
        files = [a for a in args if a != "--check"]
        if "--check" not in args or not files:
            print("Usage: apply_reviews.py [--check <split decisions file> ...]")
            return
        benchmarks = utils.load_benchmarks()
        for path in files:
            check_split_file(path, benchmarks)
        return
    matches = utils.load_json(MATCHES_PATH)
    benchmarks = utils.load_benchmarks()
    overrides = utils.load_overrides()
    reviewed = utils.load_reviewed_evaluations()

    apply_matches(overrides, matches, benchmarks)
    apply_entry_reviews("new", "leaderboard", check_new, overrides, reviewed)
    apply_entry_reviews("entries", "benchmark", check_entries, overrides, reviewed)
    reviewed["accepted"] -= reviewed["excluded"]
    apply_splits(overrides, matches, benchmarks, reviewed)

    utils.write_json(utils.OVERRIDES_PATH, overrides)
    utils.write_reviewed_evaluations(reviewed)
    print(
        f"evaluations: {len(reviewed['accepted'])} accepted, {len(reviewed['excluded'])} excluded"
    )


# ==================================================================================================

if __name__ == "__main__":
    main()
