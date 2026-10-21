import argparse
import html
import json
import os
from collections import Counter
from typing import Dict, List
from urllib.parse import urlparse

from utils import load_template, paper_slug

# ==================================================================================================

# Metrics are cut off after the last metric that still fits into this length
MAX_METRICS_CHARS = 75

# ==================================================================================================


def metric_labels(fields: List[dict]) -> Dict[str, str]:
    """Map metric keys (m1, m2, ...) to their labels."""

    labels = {}
    for field in fields:
        key = str(field.get("key", ""))
        if key.startswith("m"):
            labels[key] = str(field.get("label", key)).strip() or key
    return labels


# ==================================================================================================


def collect_papers(in_dir: str) -> Dict[str, dict]:
    """Gather all benchmark results, grouped by paper.

    Returns a dict of slug -> {names, links, entries}, where entries holds one
    record per benchmark result of that paper.
    """

    papers: Dict[str, dict] = {}

    for filename in sorted(os.listdir(in_dir)):
        if not filename.lower().endswith(".json"):
            continue

        path = os.path.join(in_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] Could not read {path}: {e}")
            continue

        benchmark = str(data.get("title", "")).strip()
        if not benchmark:
            continue

        bench_data = data.get("benchmark", {})
        labels = metric_labels(bench_data.get("fields", []))

        # Results are stored best first, so the position of an item is its rank.
        # Like for the badges, only results with code are ranked.
        rank = 0
        for item in bench_data.get("items", []):
            if not isinstance(item, dict):
                continue

            code = str(item.get("c", "")).strip()
            if code:
                rank += 1

            paper = item.get("p")
            if not isinstance(paper, dict):
                continue

            name = str(paper.get("name", "")).strip()
            link = str(paper.get("link", "")).strip()
            slug = paper_slug(name, link)
            if not slug:
                continue

            metrics = []
            for key, label in labels.items():
                value = str(item.get(key, "")).strip()
                if value:
                    metrics.append((label, value))

            record = papers.setdefault(
                slug, {"names": Counter(), "links": Counter(), "entries": []}
            )
            if name:
                record["names"][name] += 1
            if link:
                record["links"][link] += 1
            record["entries"].append(
                {
                    "benchmark": benchmark,
                    "model": str(item.get("n", "")).strip(),
                    "metrics": metrics,
                    "rank": rank if code else None,
                }
            )

    return papers


# ==================================================================================================


def display_title(record: dict) -> str:
    """Return the most used name of the paper, or its link if it has no name."""

    if record["names"]:
        return record["names"].most_common(1)[0][0]
    return record["links"].most_common(1)[0][0]


# ==================================================================================================


def render_actions(record: dict) -> str:
    """Render the buttons linking to the paper itself."""

    links = [link for link, _ in record["links"].most_common()]

    buttons = []
    for i, link in enumerate(links):
        url = html.escape(link, quote=True)
        # Further publication versions of the same paper get their own button
        label = "Open paper" if i == 0 else "Other version"
        host = html.escape(urlparse(link).netloc.replace("www.", ""))
        buttons.append(
            f'<a class="btn" href="{url}" title="{host}"'
            f' target="_blank" rel="noopener noreferrer">{label}</a>'
        )

    return "\n          ".join(buttons)


# ==================================================================================================


def render_metrics(metrics: List[tuple]) -> str:
    """Render the metrics of one result, cut off when they get too long.

    Metrics are kept whole, so rather one metric less than a cut off number.
    The full list stays available as a tooltip.
    """

    shown = []
    length = 0
    for label, value in metrics:
        text = f"{label}: {value}"
        if shown and length + len(text) > MAX_METRICS_CHARS:
            break
        shown.append(text)
        length += len(text) + 3  # the ' · ' separator between two metrics

    cells = [f'<span class="metric">{html.escape(t)}</span>' for t in shown]
    if len(shown) < len(metrics):
        cells.append('<span class="metric">&hellip;</span>')

    full = " · ".join(f"{label}: {value}" for label, value in metrics)
    title = (
        f' title="{html.escape(full, quote=True)}"' if len(shown) < len(metrics) else ""
    )
    return f'<td class="metrics"{title}>' + "".join(cells) + "</td>"


# ==================================================================================================


def render_rows(entries: List[dict]) -> str:
    """Render one table row per benchmark result of the paper."""

    # Benchmarks where the paper ranks best come first, unranked ones last
    best = {}
    for entry in entries:
        rank = entry["rank"] or float("inf")
        best[entry["benchmark"]] = min(rank, best.get(entry["benchmark"], rank))

    def sort_key(entry: dict):
        # Group by benchmark, best rank first, unranked results last
        return (
            best[entry["benchmark"]],
            entry["benchmark"],
            entry["rank"] or float("inf"),
            entry["model"],
        )

    rows = []
    for entry in sorted(entries, key=sort_key):
        benchmark = html.escape(entry["benchmark"])
        bench_link = (
            f'<a href="../benchmarks/{html.escape(entry["benchmark"], quote=True)}.html"'
            f' target="_blank" rel="noopener noreferrer">{benchmark}</a>'
        )

        rank = f'#{entry["rank"]}' if entry["rank"] else "&ndash;"

        rows.append(
            "<tr>"
            f"<td>{bench_link}</td>"
            f'<td>{html.escape(entry["model"])}</td>'
            f'<td class="rank">{rank}</td>'
            f'{render_metrics(entry["metrics"])}'
            "</tr>"
        )

    return "\n            ".join(rows)


# ==================================================================================================


def write_paper_pages(papers: Dict[str, dict], out_dir: str, template: str) -> None:
    """Write one page per paper."""

    template = template.replace("<!--[basestyle]-->", load_template("base_style.html"))

    for slug, record in papers.items():
        page = template.replace("<!--[title]-->", html.escape(display_title(record)))
        page = page.replace("<!--[actions]-->", render_actions(record))
        page = page.replace("<!--[rows]-->", render_rows(record["entries"]))

        with open(os.path.join(out_dir, slug + ".html"), "w", encoding="utf-8") as f:
            f.write(page)


# ==================================================================================================


def write_index(papers: Dict[str, dict], out_dir: str, template: str) -> None:
    """Write the paper search, which renders the matches of a query in the browser.

    The papers are handed over as data, because building thousands of list items
    into the page would only bloat a list nobody scrolls through.
    """

    items = []
    for slug, record in papers.items():
        count = len({e["benchmark"] for e in record["entries"]})
        items.append((display_title(record), slug, count))

    items.sort(key=lambda x: (x[0].lower(), x[1]))

    content = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    # The data sits in a script tag, so a '<' of a title must not end it early
    content = content.replace("<", "\\u003c")

    page = template.replace("<!--[content]-->", content)
    page = page.replace("<!--[basestyle]-->", load_template("base_style.html"))
    page = page.replace("<!--[herostyle]-->", load_template("hero_style.html"))
    page = page.replace("<!--[modal]-->", load_template("paper_modal.html"))
    page = page.replace("<!--[search]-->", load_template("search_filter.html"))

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build one HTML page per paper, plus the paper search page."
    )
    parser.add_argument("--input", "-i", default="dataset/benchmarks")
    parser.add_argument("--output", "-o", default="public/papers")
    args = parser.parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)

    if not os.path.isdir(in_dir):
        raise SystemExit(f"Input folder not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    paper_template = load_template("paper_plain.html")
    index_template = load_template("papers.html")
    papers = collect_papers(in_dir)
    write_paper_pages(papers, out_dir, paper_template)
    write_index(papers, out_dir, index_template)

    print(f"[info] Generated {len(papers)} paper pages in {out_dir}")


# ==================================================================================================

if __name__ == "__main__":
    main()
