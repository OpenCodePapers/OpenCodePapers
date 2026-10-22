"""
Download the details of papers without arXiv id (for their links), and of the papers
that external results were taken from, to data/papers/.
"""

import os
import re

import hfpwc_api
import hfpwc_utils as utils

# ==================================================================================================

MATCHES_PATH = "data/matches.json"
PAPERS_DIR = "data/papers/"

# ==================================================================================================


def paper_path(ref: str) -> str:
    return os.path.join(PAPERS_DIR, re.sub(r"[^A-Za-z0-9.]", "_", ref) + ".json")


def needed_papers(evaluations: list, matches: dict) -> set:
    refs = set()
    for e in evaluations:
        match = matches.get(utils.leaderboard_key(e), {})
        if match.get("status") not in ("merge", "new", "review", "split"):
            continue
        if not utils.is_importable(e) or utils.leaderboard_source(e):
            continue
        reporting = utils.reporting_paper_ref(e)
        if reporting:
            refs.add(reporting)
        elif not e.get("paper_arxiv_id"):
            refs.add(e["paper_id"])
    return refs


# ==================================================================================================


def main():
    _, _, evaluations = utils.load_catalog()
    matches = utils.load_json(MATCHES_PATH)
    refs = sorted(needed_papers(evaluations, matches))
    todo = [r for r in refs if not os.path.isfile(paper_path(r))]
    print(f"Papers: {len(refs)}, to download: {len(todo)}")

    for i, ref in enumerate(todo):
        data = hfpwc_api.get(f"/papers/{ref}", {"include_resources": "true"})
        utils.write_json(paper_path(ref), data or {}, compact=True)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(todo)}", flush=True)
    print("Done.")


# ==================================================================================================

if __name__ == "__main__":
    main()
