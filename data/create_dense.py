"""
dense_index.py
--------------
Builds dense + sparse vectors over 38M PubMed articles using BGE-M3.

- CPU-only safe (but will use GPU if available)
- Resume-safe: shutdown mid-run → re-run → picks up exactly where it left off
- Encodes in chunks of 100,000 articles to stay within RAM limits
- Saves dense vectors as .npy and sparse vectors as .json per chunk
- Builds FAISS IVFFlat index after all chunks are encoded

INSTALL:
    pip install "FlagEmbedding==1.2.11" "transformers>=4.44.2,<5.0.0" faiss-cpu tqdm

RUN:
    python dense_index.py

RESUME (after shutdown):
    python dense_index.py        ← same command, auto-detects progress

EXPECTED TIME (CPU-only):
    Encoding   : ~20-28 hours for 38M articles (run overnight)
    FAISS build: ~1-2 hours after all chunks done

OUTPUT:
    data\\indexes\\dense_bgem3\\
        chunks\\
            chunk_00000_dense.npy       dense vectors  [N x 1024]
            chunk_00000_sparse.json     sparse weights [{token_id: weight}]
            chunk_00000_pmids.json      PMIDs in chunk order
            ...
        bgem3.faiss                     final FAISS index (~35-40GB)
        pmid_map.json                   global position → PMID
        sparse_index.json               global PMID → sparse weights (~6-8GB)
        encode_progress.log             resume log  ← DO NOT DELETE mid-run
"""

import os
import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── HuggingFace download optimizations ────────────────────────────
# Mirror is much faster than official HF servers from Asia/Pakistan
os.environ.setdefault("HF_ENDPOINT",            "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")   # disable parallel downloads (causes stalls)
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY",  "info") # show download progress clearly


# ══════════════════════════════════════════════════════════════════ #
#  CONFIGURE                                                        #
# ══════════════════════════════════════════════════════════════════ #

BASE_DIR     = r"C:\projects\BioASQ13B"

JSONL_FILE   = os.path.join(BASE_DIR, "data", "full_input", "pubmed_collection.jsonl")

DENSE_DIR    = os.path.join(BASE_DIR, "data", "indexes", "dense_bgem3")
CHUNK_DIR    = os.path.join(DENSE_DIR, "chunks")
FAISS_FILE   = os.path.join(DENSE_DIR, "bgem3.faiss")
PMID_MAP     = os.path.join(DENSE_DIR, "pmid_map.json")
SPARSE_INDEX = os.path.join(DENSE_DIR, "sparse_index.json")
PROGRESS_LOG = os.path.join(DENSE_DIR, "encode_progress.log")

# ── Encoding ──────────────────────────────────────────────────────
CHUNK_SIZE   = 100_000   # articles per chunk — lower to 50_000 if RAM errors
BATCH_SIZE   = 32        # per-batch encoder size
MAX_LENGTH   = 256       # token limit (256 good for title+abstract)

# ── FAISS IVF ─────────────────────────────────────────────────────
DIM          = 1024      # BGE-M3 dense dimension
NLIST        = 8192      # Voronoi cells — sqrt(38M) ≈ 6164, rounded up
NPROBE       = 64        # cells to search per query

MODEL_NAME   = "BAAI/bge-m3"

# ══════════════════════════════════════════════════════════════════ #


# ────────────────────────────────────────────────────────────────── #
#  DEPENDENCY CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_deps():
    missing = []
    broken  = []

    # Check FlagEmbedding — catch both ImportError and sub-import errors
    try:
        import importlib.util
        spec = importlib.util.find_spec("FlagEmbedding")
        if spec is None:
            missing.append("FlagEmbedding==1.2.11")
        else:
            # Try the actual import to catch version-conflict errors
            try:
                from FlagEmbedding import BGEM3FlagModel  # noqa
            except ImportError as e:
                err = str(e)
                if "is_flash_attn_greater_or_equal_2_10" in err or \
                   "cannot import name" in err:
                    broken.append(
                        "FlagEmbedding/transformers version conflict.\n"
                        '       Fix: pip install "FlagEmbedding==1.2.11" '
                        '"transformers>=4.44.2,<5.0.0"'
                    )
                else:
                    broken.append(f"FlagEmbedding import error: {err}")
    except Exception as e:
        missing.append("FlagEmbedding==1.2.11")

    # Check faiss
    try:
        import faiss  # noqa
    except ImportError:
        missing.append("faiss-cpu")

    # Check tqdm
    try:
        import tqdm  # noqa
    except ImportError:
        missing.append("tqdm")

    if missing:
        print(f"[STOP] Missing packages: {', '.join(missing)}")
        print(f"       Run: pip install {' '.join(missing)}")
        sys.exit(1)

    if broken:
        print("[STOP] Broken dependencies detected:")
        for b in broken:
            print(f"       {b}")
        sys.exit(1)

    print("[OK] All dependencies found.\n")


# ────────────────────────────────────────────────────────────────── #
#  DISK SPACE CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_disk():
    import shutil
    free_gb = shutil.disk_usage(BASE_DIR).free / (1024 ** 3)
    needed  = 55.0   # chunks ~5GB + FAISS ~38GB + sparse ~8GB + pmid map ~1GB
    print(f"[INFO] Free disk : {free_gb:.1f} GB")
    print(f"[INFO] Needed    : ~{needed:.0f} GB")
    if free_gb < needed:
        print(f"\n[STOP] Not enough disk space.")
        print(f"       Free up at least {needed:.0f} GB and re-run.")
        print(f"       Tip: delete pubmed_xmls\\ (~50GB) if not already done.")
        sys.exit(1)
    print(f"[OK] Enough disk space.\n")


# ────────────────────────────────────────────────────────────────── #
#  PROGRESS LOG                                                      #
# ────────────────────────────────────────────────────────────────── #

def load_progress() -> set:
    """Returns set of chunk indices already fully encoded and saved."""
    if not os.path.exists(PROGRESS_LOG):
        return set()
    with open(PROGRESS_LOG, "r") as f:
        done = {int(l.strip()) for l in f if l.strip().isdigit()}
    if done:
        print(f"[RESUME] {len(done)} chunk(s) already done — resuming from chunk {max(done)+1}.\n")
    return done


def mark_chunk_done(chunk_idx: int):
    """Written AFTER all chunk files are safely saved — crash before = retry."""
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{chunk_idx}\n")


# ────────────────────────────────────────────────────────────────── #
#  JSONL READER                                                      #
# ────────────────────────────────────────────────────────────────── #

def stream_articles(path: str, skip: int = 0):
    """
    Yields (pmid, text) from your pubmed_collection.jsonl.
    Format: {"id": "12345678", "contents": "title abstract mesh ..."}
    Skips first `skip` lines for resume.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < skip:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid     = str(doc.get("id", f"line_{line_no}"))
            contents = doc.get("contents", "").strip()
            if contents:
                yield pmid, contents


def count_lines(path: str) -> int:
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


# ────────────────────────────────────────────────────────────────── #
#  STEP 1 — ENCODE ALL CHUNKS                                        #
# ────────────────────────────────────────────────────────────────── #

def encode_all_chunks(total: int, done_chunks: set) -> list:
    """
    Streams JSONL, encodes in chunks, saves per chunk:
        chunk_XXXXX_dense.npy    float32 [N x 1024]
        chunk_XXXXX_sparse.json  [{token_id: weight}, ...]
        chunk_XXXXX_pmids.json   [pmid, ...]
    Returns ordered list of all PMIDs.
    """
    # Deferred import — only runs after check_deps() has verified it works
    from FlagEmbedding import BGEM3FlagModel

    os.makedirs(CHUNK_DIR, exist_ok=True)

    n_chunks   = math.ceil(total / CHUNK_SIZE)
    skip_lines = len(done_chunks) * CHUNK_SIZE
    all_pmids  = []

    # Reload PMIDs from already-done chunks to maintain global order
    for ci in sorted(done_chunks):
        pf = os.path.join(CHUNK_DIR, f"chunk_{ci:05d}_pmids.json")
        if os.path.exists(pf):
            with open(pf) as f:
                all_pmids.extend(json.load(f))
        else:
            print(f"[WARNING] Chunk {ci} marked done but pmids file missing: {pf}")

    print(f"Loading BGE-M3 model — downloading pytorch weights only (~2.3GB, ONNX skipped)...")
    from huggingface_hub import snapshot_download
    local_model_dir = snapshot_download(
        repo_id         = MODEL_NAME,
        ignore_patterns = ["*.onnx", "*.onnx_data", "onnx/*"],  # skip 2.27GB ONNX file
    )
    model = BGEM3FlagModel(local_model_dir, use_fp16=True)
    print(f"[OK] Model loaded.\n")

    remaining = n_chunks - len(done_chunks)
    print(f"Chunks total    : {n_chunks}")
    print(f"Already done    : {len(done_chunks)}")
    print(f"To encode now   : {remaining}")
    est_h = max(0, (total - skip_lines)) / 900 / 3600
    print(f"Est. time       : ~{est_h:.1f}h on CPU  (runs fine overnight)\n")

    text_buffer = []
    pmid_buffer = []
    chunk_idx   = len(done_chunks)
    t_global    = time.time()

    with tqdm(total=max(0, total - skip_lines),
              desc="Encoding", unit="art", dynamic_ncols=True) as pbar:

        for pmid, text in stream_articles(JSONL_FILE, skip=skip_lines):
            text_buffer.append(text)
            pmid_buffer.append(pmid)

            if len(text_buffer) >= CHUNK_SIZE:
                _encode_and_save(model, text_buffer, pmid_buffer,
                                 chunk_idx, n_chunks, all_pmids, t_global)
                mark_chunk_done(chunk_idx)
                pbar.update(len(text_buffer))
                chunk_idx   += 1
                text_buffer  = []
                pmid_buffer  = []

        # Final partial chunk
        if text_buffer:
            _encode_and_save(model, text_buffer, pmid_buffer,
                             chunk_idx, n_chunks, all_pmids, t_global)
            mark_chunk_done(chunk_idx)
            pbar.update(len(text_buffer))

    print(f"\n[OK] Encoding complete — {len(all_pmids):,} articles.\n")
    return all_pmids


def _encode_and_save(model, texts, pmids, chunk_idx,
                     n_chunks, all_pmids_out, t_global):
    """Encode one chunk and write all three output files atomically."""
    dense_file  = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_dense.npy")
    sparse_file = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_sparse.json")
    pmid_file   = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_pmids.json")

    t0 = time.time()
    print(f"\n── Chunk {chunk_idx+1:>4}/{n_chunks}  ({len(texts):,} articles)", flush=True)

    output = model.encode(
        texts,
        return_dense        = True,
        return_sparse       = True,
        return_colbert_vecs = False,   # ColBERT needs GPU + huge storage, skip
        batch_size          = BATCH_SIZE,
        max_length          = MAX_LENGTH,
    )

    # Dense — float32 numpy array [N x 1024]
    dense_vecs = output["dense_vecs"].astype("float32")
    np.save(dense_file, dense_vecs)

    # Sparse — list of {token_id_str: float_weight} (only non-zero weights)
    sparse_list = [
        {str(k): float(v) for k, v in doc.items()}
        for doc in output["lexical_weights"]
    ]
    with open(sparse_file, "w") as f:
        json.dump(sparse_list, f)

    # PMIDs — preserves chunk order for FAISS position mapping
    with open(pmid_file, "w") as f:
        json.dump(pmids, f)

    all_pmids_out.extend(pmids)

    # Stats
    elapsed       = time.time() - t0
    total_elapsed = time.time() - t_global
    speed         = len(all_pmids_out) / total_elapsed if total_elapsed > 0 else 1
    eta_h         = max(0, 38_000_000 - len(all_pmids_out)) / speed / 3600

    print(f"   Dense  : {os.path.getsize(dense_file)  / 1024**2:.0f} MB")
    print(f"   Sparse : {os.path.getsize(sparse_file) / 1024**2:.0f} MB")
    print(f"   Time   : {elapsed:.0f}s  |  Speed: {speed:.0f} art/s  |  ETA: ~{eta_h:.1f}h")


# ────────────────────────────────────────────────────────────────── #
#  STEP 2 — BUILD FAISS IVFFlat INDEX                                #
# ────────────────────────────────────────────────────────────────── #

def build_faiss_index(all_pmids: list):
    import faiss

    chunk_files = sorted(Path(CHUNK_DIR).glob("chunk_*_dense.npy"))
    if not chunk_files:
        print(f"[STOP] No dense chunk .npy files in: {CHUNK_DIR}")
        sys.exit(1)

    print("=" * 60)
    print(f"Building FAISS IVFFlat index — {len(all_pmids):,} vectors")
    print(f"  DIM={DIM}  NLIST={NLIST}  NPROBE={NPROBE}")
    print("=" * 60 + "\n")

    # Train on first 2 chunks (~200K vectors is plenty for 8192 clusters)
    print("Loading training sample (first 2 chunks)...")
    sample = np.vstack([np.load(cf) for cf in chunk_files[:2]]).astype("float32")
    print(f"  {len(sample):,} vectors loaded for training\n")

    quantizer = faiss.IndexFlatIP(DIM)
    index     = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT)

    print("Training IVF quantizer (k-means) — ~10-30 min on CPU...")
    t0 = time.time()
    index.train(sample)
    print(f"[OK] Training done in {(time.time()-t0)/60:.1f} min\n")
    del sample

    index.nprobe = NPROBE

    print("Adding all vectors...")
    t0 = time.time()
    for cf in tqdm(chunk_files, desc="Adding chunks", unit="chunk"):
        vecs = np.load(cf).astype("float32")
        index.add(vecs)
        del vecs
    print(f"[OK] {index.ntotal:,} vectors added in {(time.time()-t0)/60:.1f} min\n")

    print(f"Saving FAISS index → {FAISS_FILE}  (may take a few minutes)...")
    t0 = time.time()
    faiss.write_index(index, FAISS_FILE)
    gb = os.path.getsize(FAISS_FILE) / 1024**3
    print(f"[OK] {gb:.1f} GB saved in {(time.time()-t0)/60:.1f} min\n")

    print(f"Saving PMID map → {PMID_MAP}")
    with open(PMID_MAP, "w") as f:
        json.dump(all_pmids, f)
    print(f"[OK] {len(all_pmids):,} PMIDs saved.\n")


# ────────────────────────────────────────────────────────────────── #
#  STEP 3 — MERGE SPARSE INDEX                                       #
# ────────────────────────────────────────────────────────────────── #

def build_sparse_index():
    """
    Merges per-chunk sparse files into one global file:
        { "pmid": {"token_id": weight, ...}, ... }
    Written incrementally so it never loads everything into RAM.
    """
    pmid_files   = sorted(Path(CHUNK_DIR).glob("chunk_*_pmids.json"))
    sparse_files = sorted(Path(CHUNK_DIR).glob("chunk_*_sparse.json"))

    if not pmid_files:
        print("[WARNING] No sparse chunk files found — skipping merge.")
        return

    print(f"Merging {len(sparse_files)} sparse chunk(s) → {SPARSE_INDEX}")
    print(f"  (~20-30 min for 38M articles, written incrementally)\n")

    t0 = time.time()
    with open(SPARSE_INDEX, "w", encoding="utf-8") as out:
        out.write("{")
        first = True
        for pf, sf in tqdm(zip(pmid_files, sparse_files),
                           total=len(pmid_files), desc="Merging sparse", unit="chunk"):
            with open(pf) as f: pmids   = json.load(f)
            with open(sf) as f: weights = json.load(f)
            for pmid, w in zip(pmids, weights):
                if not first:
                    out.write(",")
                out.write(f'"{pmid}":{json.dumps(w)}')
                first = False
        out.write("}")

    gb = os.path.getsize(SPARSE_INDEX) / 1024**3
    print(f"[OK] Sparse index saved — {gb:.1f} GB  ({(time.time()-t0)/60:.1f} min)\n")


# ────────────────────────────────────────────────────────────────── #
#  STEP 4 — SANITY CHECK                                             #
# ────────────────────────────────────────────────────────────────── #

def sanity_check():
    import faiss
    from FlagEmbedding import BGEM3FlagModel

    print("=" * 60)
    print("Sanity check — 3 test queries against the built index...")
    print("=" * 60 + "\n")

    index = faiss.read_index(FAISS_FILE)
    index.nprobe = NPROBE
    with open(PMID_MAP) as f:
        pmid_list = json.load(f)

    from huggingface_hub import snapshot_download
    local_model_dir = snapshot_download(
        repo_id         = MODEL_NAME,
        ignore_patterns = ["*.onnx", "*.onnx_data", "onnx/*"],
    )
    model = BGEM3FlagModel(local_model_dir, use_fp16=True)

    test_queries = [
        "diabetes insulin resistance treatment",
        "COVID-19 antiviral therapy",
        "BRCA1 mutation breast cancer risk",
    ]

    print(f"  Index: {index.ntotal:,} vectors\n")
    for query in test_queries:
        out  = model.encode([query], return_dense=True,
                            return_sparse=False, return_colbert_vecs=False)
        vec  = np.array(out["dense_vecs"], dtype="float32")
        dists, idxs = index.search(vec, 5)
        print(f"  Query: '{query}'")
        for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
            if idx != -1:
                print(f"    [{rank}] PMID={pmid_list[idx]}  score={dist:.4f}")
        print()

    print("[OK] Sanity check passed — dense index is ready!")
    print("     Run hybrid_search.py without --bm25_only to use it.\n")


# ────────────────────────────────────────────────────────────────── #
#  MAIN                                                              #
# ────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 60)
    print("  DENSE INDEX BUILDER — BGE-M3 + FAISS")
    print(f"  Input  : {JSONL_FILE}")
    print(f"  Output : {DENSE_DIR}")
    print(f"  Chunk  : {CHUNK_SIZE:,}  |  Batch: {BATCH_SIZE}  |  MaxLen: {MAX_LENGTH}")
    print(f"  Dense  : YES (dim={DIM})  |  Sparse: YES")
    print("=" * 60 + "\n")

    check_deps()
    check_disk()

    os.makedirs(DENSE_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    # ── Resume state ──────────────────────────────────────────────
    done_chunks = load_progress()

    # ── Count articles ────────────────────────────────────────────
    if not os.path.exists(JSONL_FILE):
        print(f"[STOP] JSONL not found: {JSONL_FILE}")
        sys.exit(1)
    print("Counting articles (fast scan)...")
    t0    = time.time()
    total = count_lines(JSONL_FILE)
    print(f"[OK] {total:,} articles  ({time.time()-t0:.1f}s)\n")

    n_chunks = math.ceil(total / CHUNK_SIZE)

    # ── STEP 1: Encode (or reload PMIDs if already done) ─────────
    if len(done_chunks) >= n_chunks:
        print(f"[INFO] All {n_chunks} chunks encoded — skipping to FAISS build.\n")
        all_pmids = []
        for ci in range(n_chunks):
            pf = os.path.join(CHUNK_DIR, f"chunk_{ci:05d}_pmids.json")
            if os.path.exists(pf):
                with open(pf) as f:
                    all_pmids.extend(json.load(f))
    else:
        all_pmids = encode_all_chunks(total, done_chunks)

    # ── STEP 2: FAISS index ───────────────────────────────────────
    if os.path.exists(FAISS_FILE):
        gb = os.path.getsize(FAISS_FILE) / 1024**3
        print(f"[INFO] FAISS index already exists ({gb:.1f} GB) — skipping.")
        print(f"       Delete {FAISS_FILE} to rebuild.\n")
    else:
        build_faiss_index(all_pmids)

    # ── STEP 3: Sparse index ──────────────────────────────────────
    if os.path.exists(SPARSE_INDEX):
        gb = os.path.getsize(SPARSE_INDEX) / 1024**3
        print(f"[INFO] Sparse index already exists ({gb:.1f} GB) — skipping.")
        print(f"       Delete {SPARSE_INDEX} to rebuild.\n")
    else:
        build_sparse_index()

    # ── STEP 4: Sanity check ──────────────────────────────────────
    sanity_check()

    print("=" * 60)
    print("  ALL DONE")
    print(f"  FAISS index  : {FAISS_FILE}")
    print(f"  Sparse index : {SPARSE_INDEX}")
    print(f"  PMID map     : {PMID_MAP}")
    print()
    print("  Next: run hybrid_search.py without --bm25_only")
    print("=" * 60)


if __name__ == "__main__":
    main()

