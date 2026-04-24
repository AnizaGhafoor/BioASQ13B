"""
hybrid_search.py
----------------
Hybrid retrieval fusing 2 signals with Reciprocal Rank Fusion (RRF):
    1. BM25          — keyword / lexical match  (Pyserini index)
    2. BiomedBERT    — biomedical semantic match (dense_biomedbert/biomedbert.faiss)

WHY BiomedBERT over MedCPT for query encoding?
  - MedCPT uses SEPARATE query + article encoders that must be matched.
    Using MedCPT query encoder against a BiomedBERT article index = wrong
    vector space → garbage results.
  - BiomedBERT uses ONE shared encoder for both indexing and querying
    (symmetric bi-encoder) — query and article vectors live in same space.
  - Rule: query encoder MUST match the encoder used in dense_index.py.

ARCHITECTURE:
  - Query encoded with BiomedBERT-large, mean-pooled, L2-normalised
  - Matches article vectors built by dense_index.py (same model, same pooling)
  - Inner product in FAISS == cosine similarity (both sides L2-normalised)
  - RRF merges BM25 ranks + dense ranks into final ranked list

Output format is IDENTICAL to old testset_inference.py — Colab reranker
and LLaMA-3 pipeline need ZERO changes.

INSTALL:
    pip install transformers faiss-cpu torch pyserini tqdm huggingface_hub

    # GPU (optional, speeds up query encoding):
    pip install torch --index-url https://download.pytorch.org/whl/cu121

RUN (test set inference):
    python hybrid_search.py --mode test --testset 3

RUN (training — prints recall@10/100/1000 for self-evaluation):
    python hybrid_search.py --mode train

FALLBACK (dense index not built yet):
    python hybrid_search.py --mode test --testset 3 --bm25_only
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ── HuggingFace endpoint ───────────────────────────────────────────
# hf-mirror.com does NOT host microsoft/BiomedNLP-* — use official endpoint
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY",  "info")


# ══════════════════════════════════════════════════════════════════ #
#  CONFIGURE                                                        #
# ══════════════════════════════════════════════════════════════════ #

BASE_DIR     = r"C:\projects\BioASQ13B"

# ── Your existing BM25 Pyserini index (unchanged) ─────────────────
BM25_INDEX   = os.path.join(BASE_DIR, "data", "indexes", "pyserini_pubmed_full")

# ── BiomedBERT dense index built by dense_index.py ────────────────
DENSE_DIR    = os.path.join(BASE_DIR, "data", "indexes", "dense_biomedbert")
FAISS_FILE   = os.path.join(DENSE_DIR, "biomedbert.faiss")
PMID_MAP     = os.path.join(DENSE_DIR, "pmid_map.json")

# ── Queries ───────────────────────────────────────────────────────
TRAIN_FILE   = os.path.join(BASE_DIR, "data", "training", "trainining14b.json")
TEST_PATTERN = os.path.join(BASE_DIR, "data", "BioASQ-task14bPhaseA-testset{}")

# ── Output ────────────────────────────────────────────────────────
RESULTS_DIR  = os.path.join(BASE_DIR, "phaseA-BM25", "results")

# ── Retrieval settings ────────────────────────────────────────────
BM25_TOP_K   = 1000    # BM25 candidates before RRF merge
DENSE_TOP_K  = 1000    # dense candidates before RRF merge
HYBRID_TOP_K = 1000    # final docs sent to Colab reranker

# ── RRF constants — tune these; higher k = weaker signal influence ─
RRF_K_BM25   = 60
RRF_K_DENSE  = 60

# ── BM25 params from your grid search ────────────────────────────
BM25_K1 = 0.6
BM25_B  = 0.3

# ── BiomedBERT query encoder ──────────────────────────────────────
# CRITICAL: must be the same model used in dense_index.py for indexing.
# Symmetric bi-encoder — one model encodes both queries and articles.
QUERY_MODEL  = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
QUERY_MAXLEN = 512     # BiomedBERT max — use full window for queries too
                       # (queries in BioASQ can be long clinical questions)

# ══════════════════════════════════════════════════════════════════ #


# ────────────────────────────────────────────────────────────────── #
#  DEPENDENCY CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_deps():
    missing = []
    for pkg, import_name in [
        ("transformers",  "transformers"),
        ("faiss-cpu",     "faiss"),
        ("torch",         "torch"),
        ("pyserini",      "pyserini"),
        ("tqdm",          "tqdm"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[STOP] Missing packages: {', '.join(missing)}")
        print(f"       Run: pip install {' '.join(missing)}")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU detected : {name}  ({vram:.1f} GB VRAM)")
    else:
        print("[OK] No GPU — CPU mode (query encoding ~0.2s per query, fine for search)")

    print("[OK] All dependencies found.\n")


# ────────────────────────────────────────────────────────────────── #
#  LOAD BM25                                                         #
# ────────────────────────────────────────────────────────────────── #

def load_bm25():
    from pyserini.search.lucene import LuceneSearcher
    if not os.path.exists(BM25_INDEX):
        print(f"[STOP] BM25 index not found: {BM25_INDEX}")
        sys.exit(1)
    s = LuceneSearcher(BM25_INDEX)
    s.set_bm25(k1=BM25_K1, b=BM25_B)
    print(f"[OK] BM25 loaded  ({s.num_docs:,} docs)  k1={BM25_K1}  b={BM25_B}")
    return s


# ────────────────────────────────────────────────────────────────── #
#  BIOMEDBERT QUERY ENCODER                                          #
# ────────────────────────────────────────────────────────────────── #

class BiomedBERTQueryEncoder:
    """
    Encodes BioASQ queries with BiomedBERT-large.

    Uses IDENTICAL pooling + normalisation as dense_index.py:
      - Mean pooling over last hidden state (mask-aware, padding excluded)
      - L2 normalisation → inner product in FAISS == cosine similarity

    This guarantees query and article vectors live in the same space.
    Changing either pooling method or norm would silently break retrieval.
    """

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading BiomedBERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[INFO] Loading BiomedBERT model → {self.device}...")
        self.model = AutoModel.from_pretrained(model_name)

        # fp16 on GPU: halves VRAM, ~1.7x faster, negligible quality loss
        if self.device.type == "cuda":
            self.model = self.model.half()

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[OK] BiomedBERT-large query encoder ready "
              f"({'fp16 GPU' if self.device.type == 'cuda' else 'fp32 CPU'})\n")

    def _mean_pool(self, last_hidden: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mask-aware mean pooling — identical to dense_index.py.
        Padding tokens (attention_mask=0) do NOT contribute to mean.
        """
        mask_exp   = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden.float() * mask_exp, dim=1)
        sum_mask   = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask

    def encode(self, query_text: str) -> np.ndarray:
        """
        Encode a single query string → float32 numpy array [1 x 1024].
        L2-normalised to unit norm (matches article vectors in FAISS index).
        """
        encoded = self.tokenizer(
            [query_text],
            padding        = True,
            truncation     = True,
            max_length     = QUERY_MAXLEN,
            return_tensors = "pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            out  = self.model(**encoded)
            vec  = self._mean_pool(out.last_hidden_state,
                                   encoded["attention_mask"])
            vec  = F.normalize(vec, p=2, dim=-1)

        return vec.cpu().float().numpy().astype("float32")  # [1 x 1024]


# ────────────────────────────────────────────────────────────────── #
#  LOAD DENSE INDEX                                                  #
# ────────────────────────────────────────────────────────────────── #

def load_dense():
    """
    Returns (faiss_index, pmid_list, query_encoder) or
            (None, None, None) if index not built yet.
    """
    import faiss

    missing = [f for f in [FAISS_FILE, PMID_MAP] if not os.path.exists(f)]
    if missing:
        print("\n[WARNING] Dense index files not found:")
        for f in missing:
            print(f"  Missing: {f}")
        print("  Run dense_index.py first to build the BiomedBERT index.")
        print("  Falling back to BM25-only mode.\n")
        return None, None, None

    print(f"[INFO] Loading FAISS index...")
    t0    = time.time()
    index = faiss.read_index(FAISS_FILE)
    index.nprobe = 64   # matches dense_index.py NPROBE
    print(f"       {index.ntotal:,} vectors  ({time.time()-t0:.1f}s)")

    print(f"[INFO] Loading PMID map...")
    with open(PMID_MAP) as f:
        pmid_list = json.load(f)
    print(f"       {len(pmid_list):,} PMIDs mapped")

    # Load BiomedBERT query encoder (same model as used for indexing)
    encoder = BiomedBERTQueryEncoder(QUERY_MODEL)

    return index, pmid_list, encoder


# ────────────────────────────────────────────────────────────────── #
#  LOAD QUERIES                                                      #
# ────────────────────────────────────────────────────────────────── #

def load_queries(filepath: str, mode: str):
    if not os.path.exists(filepath):
        print(f"[STOP] Query file not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "questions" in data:
            question_list = data["questions"]
        elif isinstance(data, list):
            question_list = data
        else:
            question_list = [data]
    except json.JSONDecodeError:
        question_list = [json.loads(l) for l in raw.splitlines() if l.strip()]

    queries, qrels_dict = [], {}
    for item in question_list:
        qid  = str(item["id"])
        text = item.get("body", item.get("query_text", item.get("question", "")))
        queries.append({"qid": qid, "query": text})
        gold = {str(u).rstrip("/").split("/")[-1]
                for u in item.get("documents", [])}
        if gold:
            qrels_dict[qid] = gold

    print(f"[OK] {len(queries):,} queries loaded  "
          f"({len(qrels_dict)} with gold labels)  [{mode} mode]\n")
    return queries, qrels_dict


# ────────────────────────────────────────────────────────────────── #
#  SEARCH FUNCTIONS                                                  #
# ────────────────────────────────────────────────────────────────── #

def bm25_search(searcher, query_text: str) -> dict:
    """Returns {pmid: rank} — rank starts at 1."""
    hits = searcher.search(query_text, k=BM25_TOP_K)
    return {hit.docid: rank for rank, hit in enumerate(hits, start=1)}


def dense_search(faiss_index, pmid_list: list,
                 encoder: BiomedBERTQueryEncoder,
                 query_text: str) -> dict:
    """
    Encodes query with BiomedBERT (mean-pool, L2-norm) and searches FAISS.
    Inner product against L2-normalised article vectors == cosine similarity.
    Returns {pmid: rank}.
    """
    vec = encoder.encode(query_text)                        # [1 x 1024]
    distances, indices = faiss_index.search(vec, DENSE_TOP_K)
    return {
        pmid_list[idx]: rank
        for rank, idx in enumerate(indices[0], start=1)
        if idx != -1
    }


# ────────────────────────────────────────────────────────────────── #
#  RRF MERGE — BM25 + BiomedBERT dense                              #
# ────────────────────────────────────────────────────────────────── #

def rrf_merge(bm25_ranks: dict, dense_ranks: dict) -> list:
    """
    score(d) = 1/(k_bm25  + rank_bm25(d))
             + 1/(k_dense + rank_dense(d))

    Documents only in one signal still get a score from that signal alone.
    Returns [(pmid, rrf_score)] sorted descending, length HYBRID_TOP_K.
    """
    all_pmids = set(bm25_ranks) | set(dense_ranks)
    scores = {}
    for pmid in all_pmids:
        s = 0.0
        if pmid in bm25_ranks:
            s += 1.0 / (RRF_K_BM25  + bm25_ranks[pmid])
        if pmid in dense_ranks:
            s += 1.0 / (RRF_K_DENSE + dense_ranks[pmid])
        scores[pmid] = s

    return sorted(scores.items(),
                  key=lambda x: x[1], reverse=True)[:HYBRID_TOP_K]


# ────────────────────────────────────────────────────────────────── #
#  FETCH DOCUMENT CONTENTS FROM BM25 INDEX                          #
# ────────────────────────────────────────────────────────────────── #

def get_contents(searcher, pmid: str) -> str:
    try:
        return json.loads(searcher.doc(pmid).raw()).get("contents", "")
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────────── #
#  MAIN SEARCH LOOP                                                  #
# ────────────────────────────────────────────────────────────────── #

def run_hybrid(queries, bm25_searcher, faiss_index, pmid_list,
               encoder, output_file, bm25_only=False):

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    total   = len(queries)
    t0_all  = time.time()
    results = {}

    signals = "BM25 only" if bm25_only else "BM25 + BiomedBERT dense  →  RRF"
    print(f"Signals : {signals}")
    print(f"Top-k   : BM25={BM25_TOP_K}  Dense={DENSE_TOP_K}  Out={HYBRID_TOP_K}")
    print(f"RRF k   : BM25={RRF_K_BM25}  Dense={RRF_K_DENSE}")
    print(f"Output  : {output_file}\n")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(queries, 1):
            qid       = q["qid"]
            query_txt = q["query"]

            # ── Signal 1: BM25 ───────────────────────────────────
            bm25_ranks  = bm25_search(bm25_searcher, query_txt)
            dense_ranks = {}

            if not bm25_only and faiss_index is not None:
                # ── Signal 2: BiomedBERT dense ────────────────────
                dense_ranks = dense_search(
                    faiss_index, pmid_list, encoder, query_txt
                )
                ranked = rrf_merge(bm25_ranks, dense_ranks)
            else:
                # BM25-only fallback — same output format
                ranked = sorted(bm25_ranks.items(), key=lambda x: x[1])
                ranked = [(pmid, 1.0 / (RRF_K_BM25 + rank))
                          for pmid, rank in ranked[:HYBRID_TOP_K]]

            # ── Build output docs ─────────────────────────────────
            docs = [
                {
                    "id":       pmid,
                    "score":    round(float(score), 8),
                    "contents": get_contents(bm25_searcher, pmid),
                }
                for pmid, score in ranked
            ]

            # Key kept as "bm25" — Colab reranker reads this key unchanged
            out = {"id": qid, "query_text": query_txt, "bm25": docs}
            out_f.write(json.dumps(out) + "\n")
            out_f.flush()   # crash-safe — flushed after every query
            results[qid] = docs

            if i % 10 == 0 or i == total:
                elapsed  = time.time() - t0_all
                eta      = (total - i) / (i / elapsed) if elapsed > 0 else 0
                mode_tag = "BM25" if bm25_only else "hybrid"
                print(f"  [{mode_tag}] {i:>4}/{total}  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0_all
    print(f"\n[OK] {total:,} queries in {elapsed:.0f}s  "
          f"({elapsed/total:.2f}s per query)\n")
    return results


# ────────────────────────────────────────────────────────────────── #
#  EVALUATE (training mode only)                                     #
# ────────────────────────────────────────────────────────────────── #

def evaluate(results: dict, qrels_dict: dict):
    print("── Evaluation ──────────────────────────────────")
    for k in [10, 100, 1000]:
        vals = []
        for qid, docs in results.items():
            gold = qrels_dict.get(qid, set())
            if not gold:
                continue
            hits = {d["id"] for d in docs[:k]} & gold
            vals.append(len(hits) / len(gold))
        if vals:
            print(f"  recall@{k:<6} {sum(vals)/len(vals):.4f}  "
                  f"({len(vals)} queries)")
    print()


# ────────────────────────────────────────────────────────────────── #
#  MAIN                                                              #
# ────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["train", "test"], default="test")
    parser.add_argument("--testset",   type=int, default=3)
    parser.add_argument("--bm25_only", action="store_true",
                        help="BM25 only — use while dense index is still building")
    args = parser.parse_args()

    print("=" * 60)
    print("  HYBRID RETRIEVAL — BM25 + BiomedBERT-large Dense + RRF")
    print(f"  Dense model : {QUERY_MODEL}")
    print(f"  Mode        : {args.mode}")
    print(f"  Top-k       : BM25={BM25_TOP_K}  Dense={DENSE_TOP_K}"
          f"  →  out={HYBRID_TOP_K}")
    print(f"  RRF k       : BM25={RRF_K_BM25}  Dense={RRF_K_DENSE}")
    print(f"  BM25-only   : {args.bm25_only}")
    print("=" * 60 + "\n")

    check_deps()

    # ── Load BM25 ────────────────────────────────────────────────
    bm25_searcher = load_bm25()

    # ── Load dense index + BiomedBERT query encoder ──────────────
    faiss_index = pmid_list = encoder = None

    if not args.bm25_only:
        faiss_index, pmid_list, encoder = load_dense()
        if faiss_index is None:
            print("[INFO] Falling back to BM25-only.\n")
            args.bm25_only = True

    # ── Query file + output path ──────────────────────────────────
    if args.mode == "train":
        query_file  = TRAIN_FILE
        output_file = os.path.join(RESULTS_DIR, "hybrid_train_results.jsonl")
    else:
        query_file  = TEST_PATTERN.format(args.testset)
        tag         = "BM25only" if args.bm25_only else "hybrid_biomedbert"
        output_file = os.path.join(
            RESULTS_DIR,
            f"{HYBRID_TOP_K}_BioASQ14b_testset{args.testset}_{tag}_results.jsonl"
        )

    queries, qrels_dict = load_queries(query_file, args.mode)

    # ── Run search ────────────────────────────────────────────────
    results = run_hybrid(
        queries       = queries,
        bm25_searcher = bm25_searcher,
        faiss_index   = faiss_index,
        pmid_list     = pmid_list,
        encoder       = encoder,
        output_file   = output_file,
        bm25_only     = args.bm25_only,
    )

    # ── Evaluate if training mode ─────────────────────────────────
    if args.mode == "train" and qrels_dict:
        evaluate(results, qrels_dict)

    print("=" * 60)
    print("  DONE")
    print(f"  Output → {output_file}")
    print()
    print("  Next: upload this file to Colab for reranking")
    print("  (format identical to your previous BM25 output)")
    print("=" * 60)


if __name__ == "__main__":
    main()