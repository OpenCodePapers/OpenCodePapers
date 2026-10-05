import json
import os

# ==================================================================================================

benchmark_dir = "../../dataset/benchmarks/"

# ==================================================================================================


def process_datasets(data: list):
    results = {}
    for item in data:
        for v in item["variants"]:
            url = ""
            paper = ""

            if item["homepage"] != "":
                url = item["homepage"]
            elif "paper" in item and item["paper"] is not None:
                if "url" in item["paper"]:
                    if "paperswithcode.com" not in item["paper"]["url"]:
                        url = item["paper"]["url"]

            if "paper" in item and item["paper"] is not None:
                if "title" in item["paper"]:
                    paper = item["paper"]["title"]

            # Variant names shared by datasets with different papers are ambiguous
            if v in results and results[v]["paper"] != paper:
                paper = ""

            results[v] = {"url": url, "paper": paper}
    return results


# ==================================================================================================


def process_benchmarks(data: list, results: dict):
    for item in data:
        process_benchmarks(item.get("subtasks", []), results)
        for ds in item.get("datasets", []):
            for dl in ds["dataset_links"]:
                if "/sota/" in dl["url"]:
                    results.setdefault(dl["url"].split("/sota/")[1], ds["dataset"])
    return results


# ==================================================================================================


def main():

    # Load datasets information
    path = "data/datasets.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_ds = process_datasets(data)

    # Map benchmark names to dataset names
    path = "data/evaluation-tables.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_bm = process_benchmarks(data, {})

    for f in sorted(os.listdir(benchmark_dir)):
        if f.endswith(".json"):
            filepath = os.path.join(benchmark_dir, f)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            dataset = data_bm.get(data["title"], "")
            paper = ""
            if dataset != "" and dataset in data_ds:
                paper = data_ds[dataset].get("paper", "")
            data["dataset-info"]["paper"] = paper

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done updating benchmark files.")


# ==================================================================================================


if __name__ == "__main__":
    main()
