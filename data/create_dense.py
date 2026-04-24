"""
dense_index.py
--------------
Builds dense vectors over 38M PubMed articles using
microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract

WHY BiomedBERT over BGE-M3?
  - Trained from scratch on PubMed + PMC full text (no general web data)
  - Understands MeSH terms, gene names, drug names natively
  - Consistently outperforms general models on BioASQ benchmarks
  - Large variant (1024-dim) matches BGE-M3 dimensionality → FAISS config unchanged

ARCHITECTURE:
  - Mean-pooling over last hidden state  → dense vector [1024]
  - Vectors L2-normalised for cosine/IP search
  - FAISS IVFFlat with Inner Product metric (equivalent to cosine after normalisation)
  - No sparse output (BiomedBERT is encoder-only; add BM25 separately if needed)

INSTALL:
    pip install torch transformers faiss-cpu tqdm huggingface_hub

    # GPU (optional but 10-15x faster):
    pip install torch --index-url https://download.pytorch.org/whl/cu121

RUN:
    python dense_index.py

RESUME (after shutdown):
    python dense_index.py        ← same command, auto-detects progress

EXPECTED TIME:
    CPU-only  : ~30-40 hours for 38M articles
    GPU A100  : ~3-4 hours
    GPU RTX3090: ~8-10 hours

OUTPUT:
    data\\indexes\\dense_biomedbert\\
        chunks\\
            chunk_00000_dense.npy       dense vectors  [N x 1024]
            chunk_00000_pmids.json      PMIDs in chunk order
            ...
        biomedbert.faiss                final FAISS index (~38-42GB)
        pmid_map.json                   global position → PMID
        encode_progress.log             resume log  ← DO NOT DELETE mid-run

INPUT MODES:
    ZIP  : set USE_ZIP = True   → reads pubmed_collection.jsonl directly from .zip
    PLAIN: set USE_ZIP = False  → reads pubmed_collection.jsonl from disk
"""

import os
import sys
import json
import time
import math
import zipfile
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── HuggingFace endpoint ───────────────────────────────────────────
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY",  "info")


# ══════════════════════════════════════════════════════════════════ #
#  CONFIGURE                                                        #
# ══════════════════════════════════════════════════════════════════ #

BASE_DIR = r"C:\projects\BioASQ13B"

# ── INPUT MODE ────────────────────────────────────────────────────
# Set USE_ZIP = True  → reads JSONL directly from inside the .zip file
# Set USE_ZIP = False → reads the plain .jsonl file from disk
USE_ZIP = True   # ← CHANGE THIS

# ── ZIP input (active when USE_ZIP = True) ────────────────────────
ZIP_FILE       = os.path.join(BASE_DIR, "data", "full_input", "pubmed_collection.zip")
JSONL_IN_ZIP   = "pubmed_collection.jsonl"   # path of the .jsonl inside the zip

# ── Plain JSONL input (active when USE_ZIP = False) ───────────────
# JSONL_FILE = os.path.join(BASE_DIR, "data", "full_input", "pubmed_collection.jsonl")

DENSE_DIR    = os.path.join(BASE_DIR, "data", "indexes", "dense_biomedbert")
CHUNK_DIR    = os.path.join(DENSE_DIR, "chunks")
FAISS_FILE   = os.path.join(DENSE_DIR, "biomedbert.faiss")
PMID_MAP     = os.path.join(DENSE_DIR, "pmid_map.json")
PROGRESS_LOG = os.path.join(DENSE_DIR, "encode_progress.log")

# ── Model ─────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"

# ── Encoding ──────────────────────────────────────────────────────
CHUNK_SIZE  = 100_000   # articles per chunk — lower to 50_000 if OOM
BATCH_SIZE  = 32        # reduce to 16 on CPU or low-VRAM GPU
MAX_LENGTH  = 512       # BiomedBERT max is 512 tokens

# ── FAISS IVF ─────────────────────────────────────────────────────
DIM    = 1024    # BiomedBERT-large hidden size
NLIST  = 8192    # Voronoi cells (~sqrt(38M))
NPROBE = 64      # cells searched per query

# ══════════════════════════════════════════════════════════════════ #


# ────────────────────────────────────────────────────────────────── #
#  DEPENDENCY CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_deps():
    missing = []
    for pkg, import_name in [
        ("torch",            "torch"),
        ("transformers",     "transformers"),
        ("faiss-cpu",        "faiss"),
        ("tqdm",             "tqdm"),
        ("huggingface_hub",  "huggingface_hub"),
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
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU detected : {name}  ({vram:.1f} GB VRAM)")
        print(f"     Batch size   : {BATCH_SIZE}  (increase if VRAM > 16GB)")
    else:
        print("[OK] No GPU — running on CPU (expect ~30-40h for 38M articles)")

    print("[OK] All dependencies found.\n")


# ────────────────────────────────────────────────────────────────── #
#  DISK SPACE CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_disk():
    import shutil
    free_gb = shutil.disk_usage(BASE_DIR).free / 1024**3
    needed  = 44.0   # chunks ~5GB + FAISS ~38GB + pmid_map ~1GB
    print(f"[INFO] Free disk : {free_gb:.1f} GB")
    print(f"[INFO] Needed    : ~{needed:.0f} GB")
    if free_gb < needed:
        print(f"\n[STOP] Not enough disk space.")
        print(f"       Free up at least {needed:.0f} GB and re-run.")
        sys.exit(1)
    print(f"[OK] Enough disk space.\n")


# ────────────────────────────────────────────────────────────────── #
#  INPUT SOURCE HELPERS                                              #
# ────────────────────────────────────────────────────────────────── #

def _open_jsonl_source():
    """
    Returns an open text-mode file object for the JSONL source.

    ZIP mode  : opens the zip, seeks to the member, returns a text wrapper.
                The zip handle is kept alive via a closure — caller must close
                the returned object (it has a .close() that closes both).
    Plain mode: opens the plain .jsonl file directly.

    Usage:
        fh = _open_jsonl_source()
        for line in fh:
            ...
        fh.close()
    """
    if USE_ZIP:
        # ── ZIP path ──────────────────────────────────────────────
        zf  = zipfile.ZipFile(ZIP_FILE, "r")
        raw = zf.open(JSONL_IN_ZIP, "r")          # returns a ZipExtFile (binary)
        text_stream = io.TextIOWrapper(raw, encoding="utf-8")

        # Attach a custom close so both handles are cleaned up together
        _orig_close = text_stream.close
        def _close_both():
            try:
                _orig_close()
            finally:
                zf.close()
        text_stream.close = _close_both
        return text_stream

    else:
        # ── Plain JSONL path ──────────────────────────────────────
        # Uncomment the JSONL_FILE line in CONFIGURE above to use this branch.
        # return open(JSONL_FILE, "r", encoding="utf-8")
        raise RuntimeError(
            "USE_ZIP is False but JSONL_FILE is not set. "
            "Uncomment the JSONL_FILE line in CONFIGURE and set USE_ZIP = False."
        )


def stream_articles(skip: int = 0):
    """
    Yields (pmid, text) from the configured input source (ZIP or plain JSONL).
    Format: {"id": "12345678", "contents": "title abstract ..."}
    Skips the first `skip` lines for resume support.
    """
    fh = _open_jsonl_source()
    try:
        for line_no, line in enumerate(fh):
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
    finally:
        fh.close()


def count_lines() -> int:
    """
    Fast byte-level line count.
    ZIP mode  : reads the compressed member as raw bytes (no full decompression).
    Plain mode: reads the file in 1MB blocks.
    """
    count = 0
    if USE_ZIP:
        # Decompress on the fly — still faster than JSON parsing
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            with zf.open(JSONL_IN_ZIP, "r") as raw:
                for chunk in iter(lambda: raw.read(1 << 20), b""):
                    count += chunk.count(b"\n")
    else:
        # ── Plain JSONL path ──────────────────────────────────────
        # with open(JSONL_FILE, "rb") as f:
        #     for chunk in iter(lambda: f.read(1 << 20), b""):
        #         count += chunk.count(b"\n")
        raise RuntimeError("USE_ZIP is False but JSONL_FILE is not configured.")
    return count


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
        print(f"[RESUME] {len(done)} chunk(s) already done — "
              f"resuming from chunk {max(done)+1}.\n")
    return done


def mark_chunk_done(chunk_idx: int):
    """Appended AFTER all chunk files are safely written — crash before = retry."""
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{chunk_idx}\n")


# ────────────────────────────────────────────────────────────────── #
#  BIOMEDBERT ENCODER                                                #
# ────────────────────────────────────────────────────────────────── #

class BiomedBERTEncoder:
    """
    Wraps BiomedBERT-large for batched mean-pool encoding.
    Vectors are L2-normalised → inner product == cosine similarity.
    """

    def __init__(self, model_name: str, device: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[INFO] Loading model    : {model_name}")
        print(f"[INFO] Device           : {self.device}")
        self.model = AutoModel.from_pretrained(model_name)

        if self.device == "cuda":
            self.model = self.model.half()   # fp16 → halves VRAM, ~1.7x speed

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[OK] BiomedBERT-large loaded "
              f"({'fp16 GPU' if self.device == 'cuda' else 'fp32 CPU'})\n")

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        """Mask-aware mean pooling — padding tokens do not contribute."""
        import torch
        mask_exp   = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden.float() * mask_exp, dim=1)
        sum_mask   = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask

    def encode(self, texts: list[str], batch_size: int = 32,
               max_length: int = 512) -> np.ndarray:
        """Encode texts → float32 numpy [N x 1024], L2-normalised."""
        import torch
        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch   = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
            vecs = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            all_vecs.append(vecs.cpu().float().numpy())
        return np.vstack(all_vecs).astype("float32")


# ────────────────────────────────────────────────────────────────── #
#  STEP 1 — ENCODE ALL CHUNKS                                        #
# ────────────────────────────────────────────────────────────────── #

def encode_all_chunks(total: int, done_chunks: set) -> list:
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

    print(f"Downloading BiomedBERT-large (~1.3 GB from huggingface.co)...")
    from huggingface_hub import snapshot_download
    local_model_dir = snapshot_download(repo_id=MODEL_NAME)
    encoder = BiomedBERTEncoder(local_model_dir)

    remaining = n_chunks - len(done_chunks)
    print(f"Chunks total    : {n_chunks}")
    print(f"Already done    : {len(done_chunks)}")
    print(f"To encode now   : {remaining}")
    est_h = max(0, total - skip_lines) / 700 / 3600
    print(f"Est. time       : ~{est_h:.1f}h on CPU  (GPU ~10-15x faster)\n")

    text_buffer = []
    pmid_buffer = []
    chunk_idx   = len(done_chunks)
    t_global    = time.time()

    with tqdm(total=max(0, total - skip_lines),
              desc="Encoding", unit="art", dynamic_ncols=True) as pbar:

        for pmid, text in stream_articles(skip=skip_lines):
            text_buffer.append(text)
            pmid_buffer.append(pmid)

            if len(text_buffer) >= CHUNK_SIZE:
                _encode_and_save(encoder, text_buffer, pmid_buffer,
                                 chunk_idx, n_chunks, total,
                                 all_pmids, t_global)
                mark_chunk_done(chunk_idx)
                pbar.update(len(text_buffer))
                chunk_idx  += 1
                text_buffer = []
                pmid_buffer = []

        if text_buffer:
            _encode_and_save(encoder, text_buffer, pmid_buffer,
                             chunk_idx, n_chunks, total,
                             all_pmids, t_global)
            mark_chunk_done(chunk_idx)
            pbar.update(len(text_buffer))

    print(f"\n[OK] Encoding complete — {len(all_pmids):,} articles.\n")
    return all_pmids


def _encode_and_save(encoder, texts, pmids, chunk_idx, n_chunks, total,
                     all_pmids_out, t_global):
    dense_file = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_dense.npy")
    pmid_file  = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_pmids.json")

    t0 = time.time()
    print(f"\n── Chunk {chunk_idx+1:>4}/{n_chunks}  ({len(texts):,} articles)", flush=True)

    dense_vecs = encoder.encode(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    np.save(dense_file, dense_vecs)
    with open(pmid_file, "w") as f:
        json.dump(pmids, f)
    all_pmids_out.extend(pmids)

    elapsed       = time.time() - t0
    total_elapsed = time.time() - t_global
    speed         = len(all_pmids_out) / total_elapsed if total_elapsed > 0 else 1
    eta_h         = max(0, total - len(all_pmids_out)) / speed / 3600

    print(f"   Dense  : {os.path.getsize(dense_file) / 1024**2:.0f} MB  "
          f"| shape {dense_vecs.shape}")
    print(f"   Time   : {elapsed:.0f}s  |  Speed: {speed:.0f} art/s  "
          f"|  ETA: ~{eta_h:.1f}h")


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

    print("Loading training sample (first 2 chunks)...")
    sample = np.vstack([np.load(cf) for cf in chunk_files[:2]]).astype("float32")
    print(f"  {len(sample):,} vectors loaded for training\n")

    quantizer = faiss.IndexFlatIP(DIM)
    index     = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT)
    faiss.omp_set_num_threads(os.cpu_count())

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
#  STEP 3 — SANITY CHECK                                             #
# ────────────────────────────────────────────────────────────────── #

def sanity_check():
    import faiss

    print("=" * 60)
    print("Sanity check — 3 biomedical test queries...")
    print("=" * 60 + "\n")

    index = faiss.read_index(FAISS_FILE)
    index.nprobe = NPROBE

    with open(PMID_MAP) as f:
        pmid_list = json.load(f)

    from huggingface_hub import snapshot_download
    local_model_dir = snapshot_download(repo_id=MODEL_NAME)
    encoder = BiomedBERTEncoder(local_model_dir)

    test_queries = [
        "diabetes insulin resistance treatment metformin",
        "COVID-19 antiviral therapy remdesivir",
        "BRCA1 mutation breast cancer risk hereditary",
    ]

    print(f"  Index : {index.ntotal:,} vectors\n")
    for query in test_queries:
        vec          = encoder.encode([query])
        dists, idxs  = index.search(vec, 5)
        print(f"  Query: '{query}'")
        for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
            if idx != -1:
                print(f"    [{rank}] PMID={pmid_list[idx]}  cosine={dist:.4f}")
        print()

    print("[OK] Sanity check passed — BiomedBERT dense index is ready!\n")


# ────────────────────────────────────────────────────────────────── #
#  MAIN                                                              #
# ────────────────────────────────────────────────────────────────── #

def main():
    src_label = (f"ZIP  → {ZIP_FILE}  [{JSONL_IN_ZIP}]"
                 if USE_ZIP else f"JSONL → (set JSONL_FILE in CONFIGURE)")

    print("=" * 60)
    print("  DENSE INDEX BUILDER — BiomedBERT-large + FAISS")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Input  : {src_label}")
    print(f"  Output : {DENSE_DIR}")
    print(f"  Chunk  : {CHUNK_SIZE:,}  |  Batch: {BATCH_SIZE}  "
          f"|  MaxLen: {MAX_LENGTH}")
    print(f"  Dense  : YES (dim={DIM}, L2-normalised)")
    print("=" * 60 + "\n")

    check_deps()
    check_disk()

    os.makedirs(DENSE_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    # Validate input file exists
    input_path = ZIP_FILE if USE_ZIP else None  # JSONL_FILE if USE_ZIP is False
    if input_path and not os.path.exists(input_path):
        print(f"[STOP] Input file not found: {input_path}")
        sys.exit(1)

    # Validate JSONL member exists inside zip
    if USE_ZIP:
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            names = zf.namelist()
            if JSONL_IN_ZIP not in names:
                print(f"[STOP] '{JSONL_IN_ZIP}' not found inside {ZIP_FILE}")
                print(f"       Members found: {names[:10]}")
                sys.exit(1)
        print(f"[OK] Found '{JSONL_IN_ZIP}' inside zip.\n")

    done_chunks = load_progress()

    print("Counting articles (fast scan)...")
    t0    = time.time()
    total = count_lines()
    print(f"[OK] {total:,} articles  ({time.time()-t0:.1f}s)\n")

    n_chunks = math.ceil(total / CHUNK_SIZE)

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

    if os.path.exists(FAISS_FILE):
        gb = os.path.getsize(FAISS_FILE) / 1024**3
        print(f"[INFO] FAISS index already exists ({gb:.1f} GB) — skipping.")
        print(f"       Delete {FAISS_FILE} to rebuild.\n")
    else:
        build_faiss_index(all_pmids)

    sanity_check()

    print("=" * 60)
    print("  ALL DONE")
    print(f"  FAISS index : {FAISS_FILE}")
    print(f"  PMID map    : {PMID_MAP}")
    print()
    print("  Next steps:")
    print("  1. Add BM25 (sparse) separately with Pyserini or Elasticsearch")
    print("  2. Hybrid re-rank: dense_score * 0.6 + bm25_score * 0.4")
    print("  3. Run hybrid_search.py")
    print("=" * 60)


if __name__ == "__main__":
    main()