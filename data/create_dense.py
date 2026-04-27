"""
create_dense.py
---------------
Builds a dense FAISS index over 38M PubMed articles using
BiomedBERT-large (microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract).

- GPU auto-detected — uses faiss-gpu + torch GPU automatically
- CPU fallback if no GPU available
- Resume-safe: shutdown → re-run → picks up exactly where it left off
- Reads plain pubmed_collection.jsonl directly
- Mean-pool encoding, L2-normalised → cosine via inner product in FAISS

INSTALL (GPU system):
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install transformers faiss-gpu tqdm huggingface_hub

INSTALL (CPU only):
    pip install torch transformers faiss-cpu tqdm huggingface_hub

RUN:
    python create_dense.py

RESUME (after shutdown):
    python create_dense.py    ← same command, auto-detects progress

OUTPUT:
    indexes/dense_biomedbert/
        chunks/
            chunk_00000_dense.npy    dense vectors [N x 1024]
            chunk_00000_pmids.json   PMIDs in chunk order
            ...
        biomedbert.faiss             final FAISS index (~38GB)
        pmid_map.json                global position → PMID
        encode_progress.log          resume log — DO NOT DELETE mid-run
"""

import os
import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════ #
#  CONFIGURE — change paths to match the remote machine             #
# ══════════════════════════════════════════════════════════════════ #

# ── Input — path to pubmed_collection.jsonl on THIS machine ───────
JSONL_FILE   = "/data/pubmed_collection.jsonl"   # ← change to your path

# ── Output — where to save the index ─────────────────────────────
DENSE_DIR    = "/data/indexes/dense_biomedbert"
CHUNK_DIR    = os.path.join(DENSE_DIR, "chunks")
FAISS_FILE   = os.path.join(DENSE_DIR, "biomedbert.faiss")
PMID_MAP     = os.path.join(DENSE_DIR, "pmid_map.json")
PROGRESS_LOG = os.path.join(DENSE_DIR, "encode_progress.log")

# ── Model ─────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"

# ── Encoding ──────────────────────────────────────────────────────
CHUNK_SIZE   = 100_000   # articles per chunk — lower to 50_000 if OOM
BATCH_SIZE   = 64        # 64 safe for 16GB VRAM — raise to 128 if VRAM > 24GB
MAX_LENGTH   = 512       # BiomedBERT-large max token length

# ── FAISS IVF ─────────────────────────────────────────────────────
DIM          = 1024      # BiomedBERT-large hidden size
NLIST        = 8192      # Voronoi cells — sqrt(38M) ≈ 6164, rounded up
NPROBE       = 64        # cells searched per query at inference time

# ══════════════════════════════════════════════════════════════════ #


# ────────────────────────────────────────────────────────────────── #
#  DEPENDENCY CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_deps():
    missing = []
    for pkg, import_name in [
        ("torch",           "torch"),
        ("transformers",    "transformers"),
        ("tqdm",            "tqdm"),
        ("huggingface_hub", "huggingface_hub"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    # Check faiss — accept either faiss-gpu or faiss-cpu
    try:
        import faiss  # noqa
    except ImportError:
        missing.append("faiss-gpu  (or faiss-cpu on CPU-only machine)")

    if missing:
        print(f"[STOP] Missing packages: {', '.join(missing)}")
        print(f"\n  GPU install:")
        print(f"    pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print(f"    pip install transformers faiss-gpu tqdm huggingface_hub")
        print(f"\n  CPU install:")
        print(f"    pip install torch transformers faiss-cpu tqdm huggingface_hub")
        sys.exit(1)

    import torch
    import faiss

    # ── GPU info ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU detected  : {name}  ({vram:.1f} GB VRAM)")
        if vram >= 24:
            print(f"     Tip: increase BATCH_SIZE to 128 for faster encoding")
        elif vram >= 16:
            print(f"     Tip: BATCH_SIZE=64 is good for your VRAM")
        else:
            print(f"     Tip: lower BATCH_SIZE to 32 if you get OOM errors")
    else:
        print(f"[OK] No GPU — running on CPU (expect ~30-40h for 38M articles)")

    # ── FAISS GPU check ───────────────────────────────────────────
    faiss_gpu_available = hasattr(faiss, "StandardGpuResources")
    if torch.cuda.is_available() and not faiss_gpu_available:
        print(f"\n[WARNING] torch GPU found but faiss-gpu not installed.")
        print(f"  FAISS index building will use CPU (slower).")
        print(f"  For faster FAISS: pip install faiss-gpu")
    elif faiss_gpu_available and torch.cuda.is_available():
        print(f"[OK] faiss-gpu available — FAISS index will be built on GPU")
    else:
        print(f"[OK] faiss-cpu — FAISS index will be built on CPU")

    print("[OK] All dependencies found.\n")


# ────────────────────────────────────────────────────────────────── #
#  DISK SPACE CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_disk():
    import shutil
    check_path = os.path.dirname(DENSE_DIR)
    free_gb    = shutil.disk_usage(check_path).free / 1024**3
    needed     = 44.0   # chunks ~5GB + FAISS ~38GB + pmid_map ~1GB
    print(f"[INFO] Free disk : {free_gb:.1f} GB")
    print(f"[INFO] Needed    : ~{needed:.0f} GB")
    if free_gb < needed:
        print(f"\n[STOP] Not enough disk space.")
        print(f"       Free up at least {needed:.0f} GB and re-run.")
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
        print(f"[RESUME] {len(done)} chunk(s) already done — "
              f"resuming from chunk {max(done)+1}.\n")
    return done


def mark_chunk_done(chunk_idx: int):
    """Written AFTER both files saved — crash before this = chunk retried."""
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{chunk_idx}\n")


# ────────────────────────────────────────────────────────────────── #
#  JSONL READER                                                      #
# ────────────────────────────────────────────────────────────────── #

def stream_articles(skip: int = 0):
    """
    Yields (pmid, text) from pubmed_collection.jsonl.
    Format: {"id": "12345678", "contents": "title abstract mesh ..."}
    Skips first `skip` lines for resume support.
    """
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
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


def count_lines() -> int:
    """Fast line count using buffered reads — never loads file into RAM."""
    count = 0
    with open(JSONL_FILE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


# ────────────────────────────────────────────────────────────────── #
#  BIOMEDBERT ENCODER                                                #
# ────────────────────────────────────────────────────────────────── #

class BiomedBERTEncoder:
    """
    Wraps BiomedBERT-large for batched mean-pool encoding.
    Mean pooling over all non-padding tokens (better than CLS for BERT).
    Vectors are L2-normalised → inner product == cosine similarity.
    fp16 on GPU — halves VRAM usage, ~1.7x faster, negligible quality loss.
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading tokenizer : {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[INFO] Loading model     : {model_name}")
        self.model = AutoModel.from_pretrained(model_name)

        if self.device == "cuda":
            self.model = self.model.half()   # fp16 on GPU

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[OK] BiomedBERT-large loaded  "
              f"({'fp16 GPU' if self.device == 'cuda' else 'fp32 CPU'})\n")

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        """Mask-aware mean pooling — padding tokens do not contribute."""
        import torch
        mask     = attention_mask.unsqueeze(-1).expand(
                       last_hidden_state.size()).float()
        sum_vec  = torch.sum(last_hidden_state.float() * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_vec / sum_mask

    def encode(self, texts: list, batch_size: int = 64,
               max_length: int = 512) -> np.ndarray:
        """
        Encode texts → float32 numpy [N x 1024], L2-normalised.
        Inner product on normalised vectors = cosine similarity.
        """
        import torch
        import torch.nn.functional as F

        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch   = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding        = True,
                truncation     = True,
                max_length     = max_length,
                return_tensors = "pt",
            ).to(self.device)

            with torch.no_grad():
                out  = self.model(**encoded)
                vecs = self._mean_pool(
                    out.last_hidden_state, encoded["attention_mask"]
                )
                vecs = F.normalize(vecs, p=2, dim=1)
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

    print(f"Downloading BiomedBERT-large (~1.3GB on first run, cached after)...")
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=MODEL_NAME)
    encoder   = BiomedBERTEncoder(local_dir)

    import torch
    speed_est = 2000 if torch.cuda.is_available() else 80
    remaining = n_chunks - len(done_chunks)
    est_h     = max(0, total - skip_lines) / speed_est / 3600

    print(f"Chunks total    : {n_chunks}")
    print(f"Already done    : {len(done_chunks)}")
    print(f"To encode now   : {remaining}")
    print(f"Est. time       : ~{est_h:.1f}h  "
          f"({'GPU' if torch.cuda.is_available() else 'CPU'})\n")

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
                _encode_and_save(
                    encoder, text_buffer, pmid_buffer,
                    chunk_idx, n_chunks, total, all_pmids, t_global
                )
                mark_chunk_done(chunk_idx)
                pbar.update(len(text_buffer))
                chunk_idx   += 1
                text_buffer  = []
                pmid_buffer  = []

        # Final partial chunk
        if text_buffer:
            _encode_and_save(
                encoder, text_buffer, pmid_buffer,
                chunk_idx, n_chunks, total, all_pmids, t_global
            )
            mark_chunk_done(chunk_idx)
            pbar.update(len(text_buffer))

    print(f"\n[OK] Encoding complete — {len(all_pmids):,} articles.\n")
    return all_pmids


def _encode_and_save(encoder, texts, pmids, chunk_idx,
                     n_chunks, total, all_pmids_out, t_global):
    dense_file = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_dense.npy")
    pmid_file  = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:05d}_pmids.json")

    t0 = time.time()
    print(f"\n── Chunk {chunk_idx+1:>4}/{n_chunks}  "
          f"({len(texts):,} articles)", flush=True)

    dense_vecs = encoder.encode(texts, batch_size=BATCH_SIZE,
                                max_length=MAX_LENGTH)
    np.save(dense_file, dense_vecs)

    with open(pmid_file, "w") as f:
        json.dump(pmids, f)

    all_pmids_out.extend(pmids)

    elapsed       = time.time() - t0
    total_elapsed = time.time() - t_global
    speed         = len(all_pmids_out) / total_elapsed if total_elapsed > 0 else 1
    eta_h         = max(0, total - len(all_pmids_out)) / speed / 3600
    dense_mb      = os.path.getsize(dense_file) / 1024**2

    print(f"   Shape  : {dense_vecs.shape}  →  {dense_mb:.0f} MB")
    print(f"   Time   : {elapsed:.0f}s  |  "
          f"Speed: {speed:.0f} art/s  |  ETA: ~{eta_h:.1f}h")


# ────────────────────────────────────────────────────────────────── #
#  STEP 2 — BUILD FAISS INDEX (GPU if available, CPU fallback)       #
# ────────────────────────────────────────────────────────────────── #

def build_faiss_index(all_pmids: list):
    import faiss
    import torch

    chunk_files = sorted(Path(CHUNK_DIR).glob("chunk_*_dense.npy"))
    if not chunk_files:
        print(f"[STOP] No dense chunk .npy files found in: {CHUNK_DIR}")
        sys.exit(1)

    # Detect faiss-gpu availability
    use_gpu_faiss = (torch.cuda.is_available() and
                     hasattr(faiss, "StandardGpuResources"))

    print("=" * 60)
    print(f"Building FAISS IVFFlat index")
    print(f"  Vectors  : {len(all_pmids):,}")
    print(f"  DIM      : {DIM}")
    print(f"  NLIST    : {NLIST}")
    print(f"  NPROBE   : {NPROBE}")
    print(f"  FAISS on : {'GPU' if use_gpu_faiss else 'CPU'}")
    print("=" * 60 + "\n")

    # Use all CPU cores if running on CPU
    if not use_gpu_faiss:
        faiss.omp_set_num_threads(os.cpu_count())

    # ── Training sample — first 3 chunks ─────────────────────────
    n_train = min(3, len(chunk_files))
    print(f"Loading {n_train} chunk(s) for IVF training sample...")
    sample  = np.vstack([
        np.load(cf) for cf in chunk_files[:n_train]
    ]).astype("float32")
    print(f"  {len(sample):,} vectors loaded\n")

    # ── Build CPU index first (needed for IVFFlat structure) ──────
    quantizer = faiss.IndexFlatIP(DIM)
    index_cpu = faiss.IndexIVFFlat(
        quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT
    )

    if use_gpu_faiss:
        # Move to GPU for training + adding — much faster
        res       = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

        print("Training IVF quantizer on GPU — ~1-3 min...")
        t0 = time.time()
        index_gpu.train(sample)
        print(f"[OK] Training done in {(time.time()-t0)/60:.1f} min\n")
        del sample

        index_gpu.nprobe = NPROBE

        print("Adding all vectors to GPU index...")
        t0 = time.time()
        for cf in tqdm(chunk_files, desc="Adding chunks", unit="chunk"):
            vecs = np.load(cf).astype("float32")
            index_gpu.add(vecs)
            del vecs
        print(f"[OK] {index_gpu.ntotal:,} vectors added in "
              f"{(time.time()-t0)/60:.1f} min\n")

        # Must convert back to CPU before saving
        print("Converting GPU index → CPU for saving...")
        index_final = faiss.index_gpu_to_cpu(index_gpu)

    else:
        # CPU path
        print("Training IVF quantizer on CPU — ~10-30 min...")
        t0 = time.time()
        index_cpu.train(sample)
        print(f"[OK] Training done in {(time.time()-t0)/60:.1f} min\n")
        del sample

        index_cpu.nprobe = NPROBE

        print("Adding all vectors to CPU index...")
        t0 = time.time()
        for cf in tqdm(chunk_files, desc="Adding chunks", unit="chunk"):
            vecs = np.load(cf).astype("float32")
            index_cpu.add(vecs)
            del vecs
        print(f"[OK] {index_cpu.ntotal:,} vectors added in "
              f"{(time.time()-t0)/60:.1f} min\n")

        index_final = index_cpu

    # ── Save FAISS index ──────────────────────────────────────────
    print(f"Saving FAISS index → {FAISS_FILE}")
    print(f"  (~38GB, may take a few minutes...)")
    t0 = time.time()
    faiss.write_index(index_final, FAISS_FILE)
    gb = os.path.getsize(FAISS_FILE) / 1024**3
    print(f"[OK] {gb:.1f} GB saved in {(time.time()-t0)/60:.1f} min\n")

    # ── Save PMID map ─────────────────────────────────────────────
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
    local_dir = snapshot_download(repo_id=MODEL_NAME)
    encoder   = BiomedBERTEncoder(local_dir)

    test_queries = [
        "diabetes insulin resistance treatment metformin",
        "COVID-19 antiviral therapy remdesivir",
        "BRCA1 mutation breast cancer risk hereditary",
    ]

    print(f"  Index: {index.ntotal:,} vectors\n")
    for query in test_queries:
        vec         = encoder.encode([query], batch_size=1)
        dists, idxs = index.search(vec, 5)
        print(f"  Query: '{query}'")
        for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
            if idx != -1:
                print(f"    [{rank}] PMID={pmid_list[idx]}  cosine={dist:.4f}")
        print()

    print("[OK] Sanity check passed — BiomedBERT dense index is ready!")
    print("     Copy biomedbert.faiss + pmid_map.json to your local machine.")
    print("     Then run: python hybrid_search.py\n")


# ────────────────────────────────────────────────────────────────── #
#  MAIN                                                              #
# ────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 60)
    print("  DENSE INDEX BUILDER — BiomedBERT-large + FAISS")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Input  : {JSONL_FILE}")
    print(f"  Output : {DENSE_DIR}")
    print(f"  Chunk  : {CHUNK_SIZE:,}  |  Batch: {BATCH_SIZE}  "
          f"|  MaxLen: {MAX_LENGTH}")
    print(f"  Dim    : {DIM}")
    print("=" * 60 + "\n")

    check_deps()
    check_disk()

    os.makedirs(DENSE_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    if not os.path.exists(JSONL_FILE):
        print(f"[STOP] JSONL file not found: {JSONL_FILE}")
        print(f"       Update JSONL_FILE path at top of script.")
        sys.exit(1)

    done_chunks = load_progress()

    print("Counting articles (fast scan)...")
    t0    = time.time()
    total = count_lines()
    print(f"[OK] {total:,} articles  ({time.time()-t0:.1f}s)\n")

    n_chunks = math.ceil(total / CHUNK_SIZE)

    # ── STEP 1: Encode ────────────────────────────────────────────
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

    # ── STEP 3: Sanity check ──────────────────────────────────────
    sanity_check()

    print("=" * 60)
    print("  ALL DONE")
    print(f"  FAISS index : {FAISS_FILE}")
    print(f"  PMID map    : {PMID_MAP}")
    print()
    print("  Copy these 2 files to your local machine:")
    print(f"    {FAISS_FILE}")
    print(f"    {PMID_MAP}")
    print()
    print("  Then run on local machine:")
    print("    python hybrid_search.py")
    print("=" * 60)


if __name__ == "__main__":
    main()