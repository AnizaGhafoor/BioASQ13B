import os
import sys
import json
from itertools import product
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════ #
#  BM25 GRID SEARCH — Pyserini  (FULL MODE — 38M articles)         #
#                                                                   #
#  Usage:                                                           #
#      python grid_search.py                                        #
# ══════════════════════════════════════════════════════════════════ #


# ------------------------------------------------------------------ #
#  CONFIGURE                                                         #
# ------------------------------------------------------------------ #

BASE_DIR     = r"C:\projects\BioASQ13B"
INDEX_PATH   = os.path.join(BASE_DIR, "data", "indexes", "pyserini_pubmed_full")
QUERIES_PATH = os.path.join (BASE_DIR, "data","training","trainining14b.json")
RESULTS_PATH = os.path.join(BASE_DIR, "phaseA-BM25", "results", "grid_search")

# ── COARSE grid (only 20 combos — runs fast, finds best region) ───
# once done, narrow K1_LIST and B_LIST around the best combo
K1_LIST = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]        # 6 values
B_LIST  = [0.1, 0.3, 0.5, 0.7, 0.9]              # 5 values  → 30 combos

# ── FINE grid (uncomment after coarse run finds best region) ──────
# example: if best was (0.8, 0.5) then fine-tune around it
# K1_LIST = [0.6, 0.7, 0.8, 0.9, 1.0]
# B_LIST  = [0.3, 0.4, 0.5, 0.6, 0.7]

NUM_RESULTS = 200
THREADS     = 2

# ------------------------------------------------------------------ #


# ══════════════════════════════════════════════════════════════════ #
#  DEPENDENCY CHECK                                                  #
# ══════════════════════════════════════════════════════════════════ #

def check_dependencies():
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa
        print("[OK] pyserini is installed.")
    except ImportError:
        print("[STOP] pyserini not found.  Run:  pip install pyserini")
        sys.exit(1)
    try:
        from ranx import Qrels, Run, evaluate  # noqa
        print("[OK] ranx is installed.\n")
    except ImportError:
        print("[STOP] ranx not found.  Run:  pip install ranx")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════ #
#  LOAD INDEX                                                        #
# ══════════════════════════════════════════════════════════════════ #

def load_index():
    from pyserini.search.lucene import LuceneSearcher
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR] Index not found: '{INDEX_PATH}'")
        sys.exit(-1)
    searcher = LuceneSearcher(INDEX_PATH)
    print(f"[OK] Pyserini index loaded: {INDEX_PATH}")
    print(f"     Total documents: {searcher.num_docs:,}\n")
    return searcher


# ══════════════════════════════════════════════════════════════════ #
#  LOAD QUERIES + QRELS                                             #
# ══════════════════════════════════════════════════════════════════ #

def get_queries(filename):
    qrels_dict   = {}
    queries      = {}
    queryid2text = {}

    if not os.path.exists(filename):
        print(f"[STOP] Queries file not found: {filename}")
        sys.exit(1)

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        all_questions = data["questions"]
        print(f"[INFO] BioASQ format — {len(all_questions):,} questions found.")
    elif isinstance(data, list):
        all_questions = data
        print(f"[INFO] JSON array format — {len(all_questions):,} questions found.")
    else:
        print("[STOP] Unrecognised format.")
        sys.exit(1)

    skipped = 0
    for question in all_questions:
        qid         = question["id"]
        query       = question["body"]
        baseline    = question.get("type", "all")
        doc_ids_raw = question.get("documents", [])

        if not doc_ids_raw:
            skipped += 1
            continue

        doc_ids = [url.rstrip("/").split("/")[-1] for url in doc_ids_raw]
        queryid2text[qid] = query

        if baseline not in queries:
            queries[baseline] = [{"qid": qid, "query": query.lower()}]
        else:
            queries[baseline].append({"qid": qid, "query": query.lower()})

        if baseline not in qrels_dict:
            qrels_dict[baseline] = {qid: {pmid: 1 for pmid in doc_ids}}
        else:
            qrels_dict[baseline].update({qid: {pmid: 1 for pmid in doc_ids}})

    total_q = sum(len(v) for v in queries.values())
    print(f"[OK] Loaded {total_q:,} queries across {len(queries)} baseline(s): {list(queries.keys())}")
    if skipped:
        print(f"[INFO] Skipped {skipped} questions with no documents.")
    print()
    return queries, qrels_dict, queryid2text


# ══════════════════════════════════════════════════════════════════ #
#  SEARCH                                                           #
# ══════════════════════════════════════════════════════════════════ #

def run_bm25(searcher, query_list, k1, b):
    searcher.set_bm25(k1=k1, b=b)
    qids      = [entry["qid"]   for entry in query_list]
    querytext = [entry["query"] for entry in query_list]
    results   = searcher.batch_search(
        queries = querytext,
        qids    = qids,
        k       = NUM_RESULTS,
        threads = THREADS,
    )
    return {
        qid: {hit.docid: hit.score for hit in hits}
        for qid, hits in results.items()
    }


# ══════════════════════════════════════════════════════════════════ #
#  EVALUATE                                                         #
# ══════════════════════════════════════════════════════════════════ #

def get_metrics(qrels_dict_baseline, run_dict):
    from ranx import Qrels, Run, evaluate
    qrels = Qrels(qrels_dict_baseline)
    run   = Run(run_dict)
    metrics = [
        "recall@200", "recall@100", "recall@10",
        "map@200",    "map@100",    "map@10",
        "ndcg@200",   "ndcg@100",  "ndcg@10",
    ]
    return evaluate(qrels, run, metrics)


# ══════════════════════════════════════════════════════════════════ #
#  WRITE RESULTS                                                    #
# ══════════════════════════════════════════════════════════════════ #

def write_results(baseline, results, results_path):
    os.makedirs(results_path, exist_ok=True)
    out_file = os.path.join(results_path, f"baseline_{baseline}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ══════════════════════════════════════════════════════════════════ #
#  AVERAGE EVALUATION                                               #
# ══════════════════════════════════════════════════════════════════ #

def calculate_average_evaluation(results_path):
    sum_scores    = defaultdict(lambda: defaultdict(float))
    num_baselines = 0

    for filename in os.listdir(results_path):
        if filename == "bm25_avg.json":
            continue
        baseline_results = os.path.join(results_path, filename)
        if not os.path.isfile(baseline_results):
            continue
        num_baselines += 1
        with open(baseline_results, "r", encoding="utf-8") as f:
            results = json.load(f)
        for params, metrics in results.items():
            for metric, score in metrics.items():
                sum_scores[params][metric] += score

    if num_baselines == 0:
        print("[WARNING] No baseline files found — skipping average.")
        return

    avg_results = {
        params: {metric: value / num_baselines for metric, value in metrics.items()}
        for params, metrics in sum_scores.items()
    }

    avg_file = os.path.join(results_path, "bm25_avg.json")
    with open(avg_file, "w", encoding="utf-8") as f:
        json.dump(avg_results, f, indent=2)

    print(f"\n[OK] Average results saved: {avg_file}")

    best_params = max(avg_results, key=lambda p: avg_results[p].get("recall@200", 0))
    best        = avg_results[best_params]
    print(f"\n  Best BM25 params (by recall@200): {best_params}")
    print(f"    recall@200 : {best.get('recall@200', 0):.4f}")
    print(f"    recall@100 : {best.get('recall@100', 0):.4f}")
    print(f"    map@200    : {best.get('map@200',    0):.4f}")
    print(f"    ndcg@200   : {best.get('ndcg@200',  0):.4f}")


# ══════════════════════════════════════════════════════════════════ #
#  MAIN                                                             #
# ══════════════════════════════════════════════════════════════════ #

def main():
    print("=" * 55)
    print("  BM25 GRID SEARCH — FULL MODE (38M articles)")
    print(f"  k1 values  : {K1_LIST}")
    print(f"  b  values  : {B_LIST}")
    print(f"  Combos     : {len(K1_LIST) * len(B_LIST)}")
    print(f"  Num results: {NUM_RESULTS}")
    print(f"  Threads    : {THREADS}")
    print("=" * 55 + "\n")

    check_dependencies()
    searcher = load_index()
    queries, qrels_dict, queryid2text = get_queries(QUERIES_PATH)

    combinations = list(product(K1_LIST, B_LIST))
    total_combos = len(combinations)

    for baseline, query_list in queries.items():

        print("=" * 55)
        print(f"  Baseline : {baseline}  |  Queries: {len(query_list)}")
        print("=" * 55)

        # ── RESUME: load existing progress ──────────────────────── #
        out_file = os.path.join(RESULTS_PATH, f"baseline_{baseline}.json")
        if os.path.isfile(out_file):
            with open(out_file, "r", encoding="utf-8") as f:
                evaluation = json.load(f)
            done = len(evaluation)
            print(f"  [RESUME] Found saved file — {done}/{total_combos} combos already done.\n")
        else:
            evaluation = {}
            print(f"  [NEW] No saved file found — starting fresh.\n")
        # ─────────────────────────────────────────────────────────── #

        for idx, (k1, b) in enumerate(combinations, start=1):
            key = str((k1, b))

            if key in evaluation:
                print(f"  [{idx:>3}/{total_combos}]  k1={k1:.1f}  b={b:.1f} ... [SKIP]")
                continue

            print(f"  [{idx:>3}/{total_combos}]  k1={k1:.1f}  b={b:.1f} ...", end=" ", flush=True)

            run_dict         = run_bm25(searcher, query_list, k1, b)
            metrics          = get_metrics(qrels_dict[baseline], run_dict)
            evaluation[key]  = metrics

            # save immediately after every combo
            write_results(baseline, evaluation, results_path=RESULTS_PATH)

            r = metrics.get("recall@200", 0)
            m = metrics.get("map@200",    0)
            print(f"recall@200={r:.4f}  map@200={m:.4f}")

        print(f"\n  [DONE] Baseline '{baseline}' complete.\n")

    calculate_average_evaluation(results_path=RESULTS_PATH)

    print("\n" + "=" * 55)
    print("  GRID SEARCH COMPLETE")
    print(f"  Results: {RESULTS_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()