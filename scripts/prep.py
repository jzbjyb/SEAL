from typing import Dict, List, Tuple, Set
import argparse
from collections import defaultdict
import os
import json
import csv
import numpy as np


def load_passages(
    path: str,
    restricted_ids: Set[str] = None,
    use_csv_reader: bool = None,
    as_numpy: bool = False,
    iterative: bool = False,
    topk: int = None,
) -> List[Tuple[str, str, str]]:  # id, text, title
    if use_csv_reader is None:
        use_csv_reader = "psgs_w100.tsv" in path
    if not os.path.exists(path):
        return
    passages = []
    with open(path) as fin:
        if use_csv_reader:
            reader = csv.reader(fin, delimiter="\t")
            header = next(reader)
        else:
            header = fin.readline().strip().split("\t")
        assert len(header) == 3 and header[0] == "id", "header format error"
        textfirst = header[1] == "text"
        for k, row in enumerate(reader if use_csv_reader else fin):
            if (k + 1) % 1000000 == 0:
                print(f"{(k + 1) // 1000000}M", end=" ", flush=True)
            try:
                if not use_csv_reader:
                    row = row.rstrip("\n").split("\t")
                if restricted_ids and row[0] not in restricted_ids:
                    continue
                if textfirst:
                    did, text, title = row[0], row[1], row[2]
                else:
                    did, text, title = row[0], row[2], row[1]
                if iterative:
                    yield did, text, title
                else:
                    passages.append((did, text, title))
            except:
                print(f"The following input line has not been correctly loaded: {row}")
            if topk is not None and len(passages) >= topk:
                break
    if not iterative:
        if as_numpy:
            yield np.array(passages, dtype=np.string_)
        yield passages


def save_beir_format(
    beir_dir: str,
    qid2dict: Dict[str, Dict] = None,
    did2dict: Dict[str, Dict] = None,
    split2qiddid: Dict[str, List[Tuple[str, str]]] = None,
):
    # save
    os.makedirs(beir_dir, exist_ok=True)
    if qid2dict is not None:
        with open(os.path.join(beir_dir, "queries.jsonl"), "w") as fout:
            for qid in qid2dict:
                fout.write(json.dumps(qid2dict[qid]) + "\n")
    if did2dict is not None:
        with open(os.path.join(beir_dir, "corpus.jsonl"), "w") as fout:
            for did in did2dict:
                fout.write(json.dumps(did2dict[did]) + "\n")
    if split2qiddid is not None:
        os.makedirs(os.path.join(beir_dir, "qrels"), exist_ok=True)
        for split in split2qiddid:
            with open(os.path.join(beir_dir, "qrels", f"{split}.tsv"), "w") as fout:
                fout.write("query-id\tcorpus-id\tscore\n")
                for qid, did in split2qiddid[split]:
                    fout.write(f"{qid}\t{did}\t1\n")


def convert_nq_to_beir_format(nq_dir: str, beir_dir: str):
    qid2dict: Dict[str, Dict] = {}
    did2dict: Dict[str, Dict] = {}
    split2qiddid: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    max_did = 0
    for did, text, title in load_passages(os.path.join(nq_dir, "psgs_w100.tsv"), iterative=True):
        did2dict[did] = {"_id": did, "title": title, "text": text}
        max_did = max(max_did, int(did))
    for split, nsplit in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        with open(os.path.join(nq_dir, f"{split}.json"), "r") as fin:
            data = json.load(fin)
            for ex in data:
                qid = f"{str(len(qid2dict) + max_did + 1)}"
                qid2dict[qid] = {"_id": qid, "text": ex["question"], "metadata": {"answer": ex["answers"]}}
                split2qiddid[nsplit].append((qid, ex["ctxs"][0]["id"]))
    save_beir_format(beir_dir, qid2dict, did2dict, split2qiddid)


def convert_dpr_to_tsv_format(dpr_file: str, tsv_file: str):
    with open(dpr_file, "r") as fin, open(tsv_file, "w") as fout:
        data = json.load(fin)
        writer = csv.writer(fout, delimiter="\t")
        for query in data:
            writer.writerow([query["question"], []])  # empty answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument("--task", type=str, choices=["convert_nq_to_beir_format", "convert_dpr_to_tsv_format"])
    parser.add_argument("--inp", type=str, help="input files/dirs", nargs="+")
    parser.add_argument("--out", type=str, help="output files/dirs", nargs="+")
    args = parser.parse_args()

    if args.task == "convert_nq_to_beir_format":
        nq_dir = args.inp[0]
        beir_dir = args.out[0]
        convert_nq_to_beir_format(nq_dir, beir_dir)
    elif args.task == "convert_dpr_to_tsv_format":
        dpr_file = args.inp[0]
        tsv_file = args.out[0]
        convert_dpr_to_tsv_format(dpr_file, tsv_file)
