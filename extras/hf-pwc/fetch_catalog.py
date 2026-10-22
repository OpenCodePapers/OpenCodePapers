"""
Download all tasks, datasets and evaluations (leaderboard entries) to data/catalog/.
"""

import json
import os

import hfpwc_api

# ==================================================================================================

CATALOG_DIR = "data/catalog/"

# ==================================================================================================


def fetch(name: str):
    results = []
    for item in hfpwc_api.get_all(f"/{name}/"):
        results.append(item)
        if len(results) % 1000 == 0:
            print(f"{name}: {len(results)}", flush=True)
    # offset pages repeat items when the site changes meanwhile
    results = list({item["id"]: item for item in results}.values())

    path = os.path.join(CATALOG_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"{name}: {len(results)} written to {path}")


# ==================================================================================================


def main():
    os.makedirs(CATALOG_DIR, exist_ok=True)
    fetch("tasks")
    fetch("datasets")
    fetch("evaluations")


# ==================================================================================================

if __name__ == "__main__":
    main()
