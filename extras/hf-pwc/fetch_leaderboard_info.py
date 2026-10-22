"""
Download the metric directions and the primary metric of the leaderboards that become
new benchmarks (also in parts, when split), to data/leaderboards/.
"""

import os

import hfpwc_api
import hfpwc_utils as utils

# ==================================================================================================

MATCHES_PATH = "data/matches.json"
LEADERBOARDS_DIR = "data/leaderboards/"

# ==================================================================================================


def info_path(key: str) -> str:
    return os.path.join(LEADERBOARDS_DIR, key.replace("/", "_") + ".json")


def fetch_info(key: str) -> dict:
    task_id, dataset_id = key.split("/")
    metrics = hfpwc_api.get(f"/datasets/{dataset_id}/tasks/{task_id}/metrics") or {}
    dataset = hfpwc_api.get(f"/datasets/{dataset_id}", {"task": task_id}) or {}
    summary = dataset.get("leaderboard_summary") or {}
    return {
        "metrics": {
            m["metric_name"]: m.get("metric_direction")
            for m in metrics.get("metrics", [])
        },
        "primary": summary.get("metric"),
    }


# ==================================================================================================


def main():
    matches = utils.load_json(MATCHES_PATH)
    keys = [
        k
        for k, m in matches.items()
        if m["status"] == "new"
        or any(g["decision"] == "new" for g in m.get("groups", []))
    ]
    todo = [k for k in keys if not os.path.isfile(info_path(k))]
    print(f"Leaderboards: {len(keys)}, to download: {len(todo)}")

    for i, key in enumerate(todo):
        utils.write_json(info_path(key), fetch_info(key))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(todo)}", flush=True)
    print("Done.")


# ==================================================================================================

if __name__ == "__main__":
    main()
