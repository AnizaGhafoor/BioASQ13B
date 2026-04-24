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
QUERIES_PATH = os.path.join(BASE_DIR, "data", "training", "trainining14b.json")

OUTPUT_FILE  = os.path.join(BASE_DIR, "phaseA-BM25", "results", "bm25_results.jsonl")

# BM25 params — best values from grid search
K1 = 0.6
B  = 0.3

# How many docs to retrieve per query
NUM_RESULTS = 1000

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
#  STEP 2 — Load queries                                            #
# ------------------------------------------------------------------ #

def get_queries(queries_path: str):
    if not os.path.exists(queries_path):
        print(f"[STOP] Queries file not found: {queries_path}")
        sys.exit(1)

    queries      = []
    qrels_dict   = {}
    queryid2text = {}

    with open(queries_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

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

    for item in question_list:
        qid  = str(item["id"])
        text = item.get("body", item.get("query_text", item.get("question", "")))

        queries.append({"qid": qid, "query": text})
        queryid2text[qid] = text

        gold = set()
        for doc_url in item.get("documents", []):
            pmid = str(doc_url).rstrip("/").split("/")[-1]
            gold.add(pmid)
        qrels_dict[qid] = gold

    print(f"[OK] Loaded {len(queries):,} queries from: {queries_path}\n")
    return queries, qrels_dict, queryid2text


# ------------------------------------------------------------------ #
#  STEP 3 — Run BM25 search                                         #
# ------------------------------------------------------------------ #

def run_search(searcher: LuceneSearcher, queries: list) -> dict:
    results = {}
    total   = len(queries)
    t0      = time.time()

    for i, q in enumerate(queries, 1):
        qid       = q["qid"]
        query_txt = q["query"]

        hits = searcher.search(query_txt, k=NUM_RESULTS)

        docs = []
        for hit in hits:
            try:
                raw      = searcher.doc(hit.docid).raw()   # fixed for newer Pyserini
                doc_json = json.loads(raw)
                contents = doc_json.get("contents", "")
            except Exception:
                contents = ""

            docs.append({
                "id":       hit.docid,
                "score":    round(float(hit.score), 6),
                "contents": contents,
            })

        results[qid] = docs

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  Searched {i:>5}/{total}  ({elapsed:.0f}s)", flush=True)

    print(f"\n[OK] Search complete — {total:,} queries in {time.time()-t0:.0f}s\n")
    return results


# ------------------------------------------------------------------ #
#  STEP 4 — Write output JSONL                                      #
# ------------------------------------------------------------------ #

def write_output(results: dict, queryid2text: dict, output_file: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for qid, docs in results.items():
            out = {
                "id":         qid,
                "query_text": queryid2text.get(qid, ""),
                "bm25":       docs,
            }
            f.write(json.dumps(out) + "\n")
            written += 1

    print(f"[OK] Results written: {written:,} queries -> {output_file}\n")


# ══════════════════════════════════════════════════════════════════ #
#  MAIN — no arguments needed, paths are hardcoded above           #
# ══════════════════════════════════════════════════════════════════ #

def main():
    print("=" * 55)
    print("  PYSERINI BM25 SEARCH")
    print(f"  Index      : {FULL_INDEX}")
    print(f"  Queries    : {QUERIES_PATH}")
    print(f"  Output     : {OUTPUT_FILE}")
    print(f"  BM25       : k1={K1}  b={B}  top={NUM_RESULTS}")
    print("=" * 55 + "\n")

    searcher                     = load_index(FULL_INDEX)
    queries, qrels_dict, id2text = get_queries(QUERIES_PATH)
    results                      = run_search(searcher, queries)
    write_output(results, id2text, OUTPUT_FILE)

    print("=" * 55)
    print("  SEARCH DONE")
    print(f"  Output -> {OUTPUT_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()