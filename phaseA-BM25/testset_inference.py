import os
import sys
import json
import time

from pyserini.search.lucene import LuceneSearcher


# ══════════════════════════════════════════════════════════════════ #
#  CONFIGURE — change these paths to match your setup              #
# ══════════════════════════════════════════════════════════════════ #

BASE_DIR     = r"C:\projects\BioASQ13B"
FULL_INDEX   = os.path.join(BASE_DIR, "data", "indexes", "pyserini_pubmed_full")

# ── Test set path — update this to your actual test file ─────────
#TESTSET_FILE = os.path.join(BASE_DIR, "data", "BioASQ-task13bPhaseA-testset1.json")
TESTSET_FILE = os.path.join(BASE_DIR,  "data", "BioASQ-task14bPhaseA-testset3")

# ── Output path ───────────────────────────────────────────────────
OUTPUT_FILE  = os.path.join(BASE_DIR, "phaseA-BM25", "results", "100_BioASQ14b_testset3_BM25_results.json")


# BM25 params
K1          = 0.6
B           = 0.3
NUM_RESULTS = 100

# ══════════════════════════════════════════════════════════════════ #


# ------------------------------------------------------------------ #
#  STEP 1 — Load Pyserini index                                     #
# ------------------------------------------------------------------ #

def load_index(index_path: str) -> LuceneSearcher:
    if not os.path.exists(index_path):
        print(f"[STOP] Index not found at: {index_path}")
        sys.exit(1)

    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=K1, b=B)
    print(f"[OK] Index loaded from: {index_path}")
    print(f"     BM25 params — k1={K1}, b={B}, num_results={NUM_RESULTS}\n")
    return searcher


# ------------------------------------------------------------------ #
#  STEP 2 — Load test set queries                                   #
#                                                                   #
#  Supports:                                                        #
#    Format 1 — BioASQ JSON  : {"questions": [{"id":..,"body":..}]} #
#    Format 2 — JSONL        : {"id":..,"body":..} per line         #
# ------------------------------------------------------------------ #

def get_queries(testset_file: str):
    """
    Load test queries — no gold documents expected.

    Returns:
        queries      : list of {"qid": str, "query": str}
        queryid2text : {qid: question_text}
    """
    if not os.path.exists(testset_file):
        print(f"[STOP] Test set file not found: {testset_file}")
        sys.exit(1)

    queries      = []
    queryid2text = {}

    with open(testset_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # ── Auto-detect format ───────────────────────────────────────
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "questions" in data:
            question_list = data["questions"]
            print(f"[INFO] Detected standard BioASQ JSON format.")
        elif isinstance(data, list):
            question_list = data
            print(f"[INFO] Detected JSON array format.")
        else:
            question_list = [data]
    except json.JSONDecodeError:
        question_list = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                question_list.append(json.loads(line))
        print(f"[INFO] Detected JSONL format.")

    # ── Parse — only id + body, no documents field needed ────────
    for item in question_list:
        qid  = str(item["id"])
        text = item.get("body", item.get("query_text", item.get("question", "")))

        queries.append({"qid": qid, "query": text})
        queryid2text[qid] = text

    print(f"[OK] Loaded {len(queries):,} test queries from: {testset_file}\n")
    return queries, queryid2text


# ------------------------------------------------------------------ #
#  STEP 3+4 — Search AND write back to back (query by query)       #
# ------------------------------------------------------------------ #

def search_and_write(searcher: LuceneSearcher, queries: list, queryid2text: dict, output_file: str):
    """
    Search each query and immediately write result to file.
    Output format matches reference testset_inference.py:
        {"id": qid, "query_text": "...", "bm25": [{"id": pmid, "score": ..., "contents": ...}, ...]}
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    total = len(queries)
    t0    = time.time()

    with open(output_file, "w", encoding="utf-8") as f:

        for i, q in enumerate(queries, 1):
            qid       = q["qid"]
            query_txt = q["query"]

            hits = searcher.search(query_txt, k=NUM_RESULTS)

            docs = []
            for hit in hits:
                try:
                    raw      = searcher.doc(hit.docid).raw()
                    doc_json = json.loads(raw)
                    contents = doc_json.get("contents", "")
                except Exception:
                    contents = ""

                docs.append({
                    "id":       hit.docid,
                    "score":    round(float(hit.score), 6),
                    "contents": contents,
                })

            # ── Write immediately after each query ───────────────
            out = {
                "id":         qid,
                "query_text": queryid2text.get(qid, ""),
                "bm25":       docs,
            }
            f.write(json.dumps(out) + "\n")
            f.flush()   # force write to disk immediately

            if i % 10 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  Searched & saved {i:>4}/{total}  ({elapsed:.0f}s)", flush=True)

    print(f"\n[OK] All done — {total:,} queries saved to: {output_file}\n")


# ══════════════════════════════════════════════════════════════════ #
#  MAIN                                                             #
# ══════════════════════════════════════════════════════════════════ #

def main():
    print("=" * 55)
    print("  PYSERINI BM25 — TEST SET INFERENCE")
    print(f"  Index      : {FULL_INDEX}")
    print(f"  Test set   : {TESTSET_FILE}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"  BM25       : k1={K1}  b={B}  top={NUM_RESULTS}")
    print("  Mode       : writing after every query (crash-safe)")
    print("=" * 55 + "\n")

    searcher             = load_index(FULL_INDEX)
    queries, queryid2text = get_queries(TESTSET_FILE)
    search_and_write(searcher, queries, queryid2text, OUTPUT_FILE)

    print("=" * 55)
    print("  INFERENCE DONE")
    print(f"  Output -> {OUTPUT_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()