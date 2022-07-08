from typing import List, Dict
import argparse
from collections import defaultdict
import json
import numpy as np


def clean_text(text: str):
    return text.replace("[ROW]", "").replace("[COL]", "").replace("[HEADER]", "")


def evaluate(
    prediction_file: str,
    gold_file: str = None,
    metrics: List[str] = ["title", "unigram", "exact"],
    topks: List[int] = [1, 10, 100],
):
    metric2topk2vals: Dict[str, Dict[int, List[float]]] = {m: defaultdict(list) for m in metrics}
    with open(prediction_file, "r") as fin, open(gold_file or prediction_file, "r") as gfin:
        data = json.load(fin)
        data_gold = json.load(gfin)
        assert len(data) == len(data_gold)
        for query, query_gold in zip(data, data_gold):
            if "title" in metrics:
                gold_titles = set([ctx["title"].replace("_", " ") for ctx in query_gold["positive_ctxs"]])
                title_topk2vals = {}
                for topk in topks:
                    pred_titles = set([ctx["title"].replace("_", " ") for ctx in query["ctxs"][:topk]])
                    title_topk2vals[topk] = float(len(pred_titles & gold_titles) > 0)
                    metric2topk2vals["title"][topk].append(title_topk2vals[topk])
            if "unigram" in metrics:
                gold_tokens_li = [set(clean_text(ctx["text"]).split()) for ctx in query_gold["positive_ctxs"]]
                unigram_topk2vals = {}
                for topk in topks:
                    pred_tokens_li = [set(ctx["text"].split()) for ctx in query["ctxs"][:topk]]
                    unigram_topk2vals[topk] = np.max(
                        [len(pt & gt) / (len(gt) or 1) for pt in pred_tokens_li for gt in gold_tokens_li]
                    )
                    metric2topk2vals["unigram"][topk].append(unigram_topk2vals[topk])
            if "exact" in metrics:
                assert (
                    "title" in metrics and "unigram" in metrics
                )  # exact means retrieving the same article (i.e., same title) and unigram overlap >= 0.9
                for topk in topks:
                    metric2topk2vals["exact"][topk].append(
                        float(title_topk2vals[topk] == 1 and unigram_topk2vals[topk] >= 0.9)
                    )

    print("metric\t" + "\t".join(map(str, topks)))  # header
    for metric, topk2vals in metric2topk2vals.items():
        vals = [topk2vals[topk] for topk in topks]
        print(f"{metric}\t" + "\t".join(str(np.mean(val)) for val in vals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=str, help="prediction file")
    parser.add_argument("--gold", type=str, default=None, help="gold file")
    args = parser.parse_args()

    evaluate(args.prediction, args.gold)
