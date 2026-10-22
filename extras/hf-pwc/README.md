# HF Papers with Code

Imports benchmark results from the [Papers with Code](https://paperswithcode.co/tasks) site
of _Hugging Face_, through its [public API](https://paperswithcode.co/docs/api).

```bash
python3 fetch_catalog.py            # tasks, datasets and evaluations (leaderboard entries)
python3 match_leaderboards.py       # existing benchmark of each leaderboard
# review data/review/matches_*.json and split_*.json
python3 apply_reviews.py && python3 match_leaderboards.py
python3 fetch_leaderboard_info.py   # metric directions of new benchmarks
python3 fetch_paper_details.py      # links of papers without arXiv id
python3 build_entries.py            # entries of existing and new benchmarks
# review data/review/new_*.json, entries_*.json and new_columns.json
python3 apply_reviews.py && python3 build_entries.py
python3 update_benchmarks.py        # --dry-run to only print the changes
```

Rules:

- Results are credited to the evaluated model's paper, or to the paper that reports them.
  Results of official test servers count as the method's own. Results of the benchmark
  leaderboards listed in `utils.py` are credited to the leaderboard (as a snapshot), other
  websites (aggregators, arenas, vendor reports) are dropped.
- A leaderboard is merged into an existing benchmark if both list the same results for
  several papers. Weaker matches are reviewed, to avoid duplicate benchmarks.
- Leaderboards that mix protocols (like zero-shot and fine-tuned results) are split by a
  review into groups of one protocol, which are merged or become new benchmarks.
- New benchmarks need at least 5 entries from 3 papers and a code link.
- Every entry is reviewed before it is added, and uncertain ones are dropped.

The details are described in the scripts.

<br>

## Reviews

Each review file in `data/review/` is checked by a reviewer (a _Claude_ sub-agent with the
instructions of [review_prompts.md](./review_prompts.md)), who writes the decisions to
`<file>.decisions.json`. They are kept in `data/overrides.json` and
`data/reviewed_evaluations.json`, so reruns only review new data (entries already in the
dataset are skipped anyway). New metric columns are decided by hand in `data/overrides.json`.

In Oct 2026, 153 of 2027 leaderboards were matched automatically and 636 reviewed. 333
leaderboards that mix protocols (zero-shot and fine-tuned results, validation and test
splits) were split by reviews, 221 of them into 287 groups of one protocol. In total, 5441
entries were added to 392 existing benchmarks, and 485 new benchmarks were created with
9435 entries. 1102 entries come from benchmark leaderboard websites.

<br>

## License

The data of Papers with Code (paperswithcode.co, by Hugging Face) is available under the
MIT License, as stated by its maintainers in
<https://github.com/huggingface/paperswithcode-feedback/issues/49>:

```text
MIT License

Copyright (c) Hugging Face

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
