"""
Add the reviewed updates of build_entries.py (data/updates/) to the benchmark files.

Entries are inserted in the order the repo test checks (first metric, then date), with
a "%" if most values of their column have one. Existing entries and benchmarks are
skipped. With --dry-run the changes are only printed.
"""

import json
import os
import re
import sys
from typing import List, Optional

import hfpwc_utils as utils

# ==================================================================================================

UPDATES_DIR = "data/updates/"

# ==================================================================================================


def first_metric_key(fields: List[dict]) -> Optional[str]:
    for f in fields:
        key = f.get("key")
        if isinstance(key, str) and key.startswith("m"):
            return key
    return None


def is_ascending(items: List[dict], key: str) -> bool:
    """Order of the existing items (median of the value steps), as in the repo test."""
    values = [v for it in items if (v := utils.metric_value(it.get(key))) is not None]
    diffs = sorted(b - a for a, b in zip(values, values[1:]) if b != a)
    return bool(diffs) and diffs[len(diffs) // 2] >= 0


def percent_metrics(items: List[dict], keys: List[str]) -> set:
    """Metrics whose existing values are mostly written with a "%" sign."""
    result = set()
    for key in keys:
        values = [str(it[key]) for it in items if it.get(key) not in (None, "")]
        if values and sum(v.strip().endswith("%") for v in values) > len(values) / 2:
            result.add(key)
    return result


def insert_position(
    items: List[dict], key: str, ascending: bool, value, date: str
) -> int:
    """Index before the first item that comes after the new one."""
    new = utils.metric_value(value)
    if new is None:
        return len(items)
    for i, item in enumerate(items):
        old = utils.metric_value(item.get(key))
        if old is None:
            continue
        if (old > new) if ascending else (old < new):
            return i
        if old == new and str(item.get("d", "")) > date:
            return i
    return len(items)


def item_key(item: dict) -> tuple:
    p = item.get("p")
    title = p.get("name", "") if isinstance(p, dict) else str(p or "")
    return (utils.normalize_title(title), utils.normalize_title(item.get("n", "")))


def merge_update(data: dict, update: dict) -> List[str]:
    """Add the new columns and entries of an update to a benchmark, return the added names."""
    fields = data["benchmark"]["fields"]
    items = data["benchmark"]["items"]
    pos = next(i for i, f in enumerate(fields) if f["key"] == "n")
    fields[pos:pos] = update["new_fields"]

    key = first_metric_key(fields)
    ascending = is_ascending(items, key)
    mkeys = [f["key"] for f in fields if f["key"].startswith("m")]
    pct_keys = percent_metrics(items, mkeys)

    existing = {item_key(it) for it in items}
    names = []
    for entry in update["entries"]:
        if item_key(entry) in existing:
            continue
        item = {k: entry[k] for k in ("p", "c", "n", "d")}
        for k in sorted(
            (k for k in entry if re.fullmatch(r"m\d+", k)), key=lambda k: int(k[1:])
        ):
            value = str(entry[k])
            item[k] = (
                f"{value}%" if k in pct_keys and not value.endswith("%") else value
            )
        index = insert_position(items, key, ascending, item.get(key), item["d"])
        items.insert(index, item)
        existing.add(item_key(item))
        names.append(f"{item['n']} ({item.get(key, '-')}) at {index}")
    return names


def write_benchmark(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))


# ==================================================================================================


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    merge_dir = os.path.join(UPDATES_DIR, "merge")
    new_dir = os.path.join(UPDATES_DIR, "new")

    added = 0
    for fname in sorted(os.listdir(merge_dir)) if os.path.isdir(merge_dir) else []:
        update = utils.load_json(os.path.join(merge_dir, fname))
        path = os.path.join(utils.BENCHMARKS_DIR, f"{update['benchmark']}.json")
        if not os.path.isfile(path):
            print(
                f"{update['benchmark']}: benchmark file not found, skipping",
                file=sys.stderr,
            )
            continue
        data = utils.load_json(path)
        names = merge_update(data, update)
        if not names:
            continue
        added += len(names)
        print(f"{update['benchmark']}: {len(names)} new")
        if not dry_run:
            write_benchmark(path, data)

    created = 0
    for fname in sorted(os.listdir(new_dir)) if os.path.isdir(new_dir) else []:
        path = os.path.join(utils.BENCHMARKS_DIR, fname)
        if os.path.exists(path):
            print(f"{fname}: benchmark already exists, skipping", file=sys.stderr)
            continue
        data = utils.load_json(os.path.join(new_dir, fname))
        created += 1
        print(
            f"{data['title']}: new benchmark with {len(data['benchmark']['items'])} entries"
        )
        if not dry_run:
            write_benchmark(path, data)

    print(
        f"Done. added={added} entries, created={created} benchmarks"
        + (" (dry run)" if dry_run else "")
    )
    return 0


# ==================================================================================================

if __name__ == "__main__":
    sys.exit(main())
