# Review prompts

Instructions for the reviewers of `data/review/` (one sub-agent per file, `<file>` is the
absolute path of the review file). Every prompt starts with the common part.

## Common part

You review data of leaderboards from the Papers with Code site (paperswithcode.co, "hf")
for the OpenCodePapers dataset. Data accuracy is much more important than quantity: when
unsure, drop or exclude. Input file: `<file>`. Write the output to
`<file without .json>.decisions.json`, one object per input element in the same order,
and validate it with `python3 -c "import json; print(len(json.load(open('<output>'))))"`.
Work only from the input files and your own knowledge, do not modify other project files,
do not browse the web, and never send the user's email address or other personal data to
any service. Put helper scripts in your own scratch subfolder. Finish with the counts of
each decision and notable problems.

## matches\_\*.json

Each element is a hf leaderboard (metrics, sample entries with "methodology" notes) with
up to 4 candidate benchmarks (metric columns m1, m2, ..., sample entries,
"shared_paper_examples" with hf values next to the benchmark values). Decide:

- `merge`: same dataset, split, version, protocol and task setting as a candidate. Map each
  hf metric with an exact counterpart: `{"key": "mX", "factor": F}`, F=1 for the same
  scale, 100 if hf uses fractions and the benchmark percent, -100 for the opposite.
  Clearly different values for the same model mean a different split or protocol.
- `new`: different from all candidates, consistent and useful on its own (num_entries >= 5,
  num_papers >= 3).
- `drop`: unsure, unclear mapping or scale, mixed protocols (zero-shot, few-shot,
  linear-probe and fine-tuned results, different input lengths), implausible values, or a
  small leaderboard that doesn't clearly match.

Candidates with another task framing (zero-shot, few-shot, real-time, domain adaptation,
1-shot vs 5-shot) or another version or split (v1 vs v2, val vs test) are different
benchmarks. Output:
`{"leaderboard", "decision", "benchmark" (merge), "metrics" (merge), "reason" (<= 20 words)}`

## new\_\*.json

Each element is a candidate new benchmark (task, dataset, metric columns with the primary
one first, proposed task hierarchy, entries with ids and methodology notes). Reference:
`data/review/task_hierarchies.json`. Decide:

- `keep`: the entries form a consistent benchmark. Exclude entries that don't fit (other
  split or setting, implausible values, other scale, duplicates); at least 5 entries from 3
  papers must remain. Give `task_hierarchy`, general to specific, reusing an existing path
  whenever one fits (extend it by one task, or create a short new path only if none fits).
  The benchmark is named `<last task>-on-<dataset>`.
- `drop`: a large part would need exclusion, unclear measure, or unreliable values.

Output: `{"leaderboard", "decision", "task_hierarchy" (keep), "exclude": [ids], "reason"}`

## entries\_\*.json

Each element is a benchmark (title, task hierarchy, metric columns, sample entries) with
`new_entries` to add (values already in the benchmark's scale). Exclude entries with
another split, setting or protocol (zero-shot or few-shot in a fine-tuning benchmark,
tools or agents, other input lengths, shots or pass@k), another metric definition,
implausible values, duplicates, or minor variants (quantized or pruned models). Settings
the benchmark already mixes may stay. Decide `keep` (with `exclude`) or `drop` if most
entries don't fit.

Output: `{"benchmark", "decision", "exclude": [ids], "reason" (<= 25 words)}`
