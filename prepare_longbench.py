"""Build LongBench (v1, English) JSONL files with the official prompt templates.

Inputs: an unpacked THUDM/LongBench data.zip and a clone of github.com/THUDM/LongBench.
Output per dataset: {out}/{dataset}.jsonl with fields
    prompt, answers, all_classes, length, _id, dataset, max_gen
and a copy of LongBench's metrics.py in {out}/ for scoring.
"""
import argparse
import json
import pathlib
import shutil

ENGLISH = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="unpacked data.zip")
    p.add_argument("--repo", required=True, help="clone of THUDM/LongBench")
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", default=",".join(ENGLISH))
    a = p.parse_args()

    repo = pathlib.Path(a.repo)
    cfgs = sorted(repo.rglob("config/dataset2prompt.json"))
    if not cfgs:
        raise SystemExit("dataset2prompt.json not found in LongBench clone")
    cfg_dir = cfgs[0].parent
    prompts = json.loads((cfg_dir / "dataset2prompt.json").read_text(encoding="utf-8"))
    maxlen = json.loads((cfg_dir / "dataset2maxlen.json").read_text(encoding="utf-8"))
    metrics = cfg_dir.parent / "metrics.py"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(metrics, out / "metrics.py")
    print("config:", cfg_dir, "metrics:", metrics)

    root = pathlib.Path(a.data_root)
    for ds in a.datasets.split(","):
        src = [f for f in root.rglob(f"{ds}.jsonl")]
        if len(src) != 1:
            raise SystemExit(f"{ds}: expected one {ds}.jsonl under {root}, found {src}")
        n = 0
        with open(src[0], encoding="utf-8") as fi, open(out / f"{ds}.jsonl", "w", encoding="utf-8") as fo:
            for line in fi:
                obj = json.loads(line)
                fo.write(json.dumps({
                    "prompt": prompts[ds].format(**obj),
                    "answers": obj["answers"],
                    "all_classes": obj.get("all_classes"),
                    "length": obj.get("length"),
                    "_id": obj.get("_id"),
                    "dataset": ds,
                    "max_gen": maxlen[ds],
                }) + "\n")
                n += 1
        print(f"{ds}: {n} rows, max_gen={maxlen[ds]}")


if __name__ == "__main__":
    main()
