
import os
import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════ #
#  CONFIGURE                                                        #
# ══════════════════════════════════════════════════════════════════ #

JSONL_FILE   = r"E:\Research Projects\BioASQ\pubmed_collection\pubmed_collection.jsonl"

DENSE_DIR    = r"E:\Research Projects\BioASQ\BioASQ13B\indexes\dense_biomedbert"
CHUNK_DIR    = os.path.join(DENSE_DIR, "chunks")
FAISS_FILE   = os.path.join(DENSE_DIR, "biomedbert.faiss")
PMID_MAP     = os.path.join(DENSE_DIR, "pmid_map.json")
PROGRESS_LOG = os.path.join(DENSE_DIR, "encode_progress.log")
COUNT_CACHE  = os.path.join(DENSE_DIR, "article_count.txt")   # ← NEW

MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"

CHUNK_SIZE        = 100_000
BATCH_SIZE        = 320
MAX_LENGTH        = 128
NUM_WORKERS       = 0
USE_DATALOADER    = False
USE_TORCH_COMPILE = False  
DIM               = 1024
NLIST             = 8192
NPROBE            = 64

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

    try:
        import faiss  # noqa
    except ImportError:
        missing.append("faiss-cpu  (or faiss-gpu)")

    if missing:
        print(f"[STOP] Missing packages: {', '.join(missing)}")
        print(f"       pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print(f"       pip install transformers faiss-cpu tqdm huggingface_hub numpy")
        sys.exit(1)

    import torch, faiss

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU : {name}  ({vram:.1f} GB VRAM)  BATCH_SIZE={BATCH_SIZE}")
    else:
        print(f"[OK] No GPU — CPU mode  (expect ~30-40h for 38M articles)")

    faiss_gpu = hasattr(faiss, "StandardGpuResources")
    if torch.cuda.is_available() and not faiss_gpu:
        print(f"[INFO] faiss-cpu detected — FAISS index build will use CPU (~30 min, fine)")
    elif faiss_gpu:
        print(f"[OK] faiss-gpu — FAISS index build on GPU")

    print("[OK] All dependencies found.\n")


# ────────────────────────────────────────────────────────────────── #
#  DISK SPACE CHECK                                                  #
# ────────────────────────────────────────────────────────────────── #

def check_disk():
    import shutil
    check_path = os.path.splitdrive(DENSE_DIR)[0] + os.sep
    free_gb    = shutil.disk_usage(check_path).free / 1024**3
    needed     = 44.0
    print(f"[INFO] Free disk : {free_gb:.1f} GB  |  Needed: ~{needed:.0f} GB")
    if free_gb < needed:
        print(f"[STOP] Not enough disk space. Free up {needed:.0f} GB and re-run.")
        sys.exit(1)
    print(f"[OK] Enough disk space.\n")


# ────────────────────────────────────────────────────────────────── #
#  ARTICLE COUNT  — cached so re-runs skip the 70-110 second scan   #
# ────────────────────────────────────────────────────────────────── #

def get_total_articles() -> int:
    """
    Returns total article count.
    First run  → counts lines (~70-110s), saves result to article_count.txt
    Every subsequent run → reads cached number instantly (0s).
    Delete article_count.txt to force a fresh count.
    """
    os.makedirs(DENSE_DIR, exist_ok=True)

    # ── Check cache first ─────────────────────────────────────────
    if os.path.exists(COUNT_CACHE):
        with open(COUNT_CACHE, "r") as f:
            cached = f.read().strip()
        if cached.isdigit():
            total = int(cached)
            print(f"[OK] Article count : {total:,}  (cached — skipping scan)\n")
            return total

    # ── Cache missing or invalid → count for real ─────────────────
    print("Counting articles (one-time scan, result will be cached)...")
    t0    = time.time()
    total = _count_lines_raw()
    elapsed = time.time() - t0
    print(f"[OK] {total:,} articles  ({elapsed:.1f}s)")

    # Save to cache
    with open(COUNT_CACHE, "w") as f:
        f.write(str(total))
    print(f"[OK] Count cached → {COUNT_CACHE}  (future runs skip this scan)\n")

    return total


def _count_lines_raw() -> int:
    """Raw buffered line count — never loads file into RAM."""
    count = 0
    with open(JSONL_FILE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


# ────────────────────────────────────────────────────────────────── #
#  PROGRESS LOG                                                      #
# ────────────────────────────────────────────────────────────────── #

def load_progress() -> set:
    if not os.path.exists(PROGRESS_LOG):
        return set()
    with open(PROGRESS_LOG, "r") as f:
        done = {int(l.strip()) for l in f if l.strip().isdigit()}
    if done:
        print(f"[RESUME] {len(done)} chunk(s) already done — "
              f"resuming from chunk {max(done)+1}.\n")
    return done


def mark_chunk_done(chunk_idx: int):
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"{chunk_idx}\n")


# ────────────────────────────────────────────────────────────────── #
#  JSONL READER                                                      #
# ────────────────────────────────────────────────────────────────── #

def stream_articles(skip: int = 0):
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


# ────────────────────────────────────────────────────────────────── #
#  BIOMEDBERT ENCODER                                                #
# ────────────────────────────────────────────────────────────────── #

class BiomedBERTEncoder:
    """
    BiomedBERT-large encoder with DataLoader parallelism.
    USE_DATALOADER=True: CPU workers tokenise next batch while
    GPU encodes current batch — hides CPU bottleneck entirely.
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading tokenizer : {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[INFO] Loading model     : {model_name}  → {self.device}")
        self.model = AutoModel.from_pretrained(model_name)

        if self.device == "cuda":
            self.model = self.model.half()   # fp16 — halves VRAM, ~1.7x faster

        self.model = self.model.to(self.device)
        self.model.eval()

        # torch.compile — extra ~10-15% on torch>=2.0, adds 4min first-batch warmup
        if USE_TORCH_COMPILE and self.device == "cuda":
            try:
                self.model = torch.compile(self.model)
                print(f"[OK] torch.compile enabled (first batch warmup is slow — normal)")
            except Exception as e:
                print(f"[INFO] torch.compile skipped: {e}")

        print(f"[OK] BiomedBERT-large ready  "
              f"({'fp16 GPU' if self.device == 'cuda' else 'fp32 CPU'})\n")

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        import torch
        mask     = attention_mask.unsqueeze(-1).expand(
                       last_hidden_state.size()).float()
        sum_vec  = torch.sum(last_hidden_state.float() * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_vec / sum_mask

    def encode(self, texts: list, batch_size: int = 128,
               max_length: int = 256) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        class _TextDataset(Dataset):
            def __init__(self, texts): self.texts = texts
            def __len__(self):         return len(self.texts)
            def __getitem__(self, i):  return self.texts[i]

        def _collate(batch_texts):
            return self.tokenizer(
                batch_texts,
                padding        = True,
                truncation     = True,
                max_length     = max_length,
                return_tensors = "pt",
            )

        all_vecs = []

        if USE_DATALOADER and self.device == "cuda":
            loader = DataLoader(
                _TextDataset(texts),
                batch_size      = batch_size,
                collate_fn      = _collate,
                num_workers     = NUM_WORKERS,
                pin_memory      = True,   # faster CPU→GPU transfer
                prefetch_factor = 2,      # pre-load 2 batches ahead
            )
            for encoded in loader:
                encoded = {k: v.to(self.device, non_blocking=True)
                           for k, v in encoded.items()}
                with torch.no_grad():
                    out  = self.model(**encoded)
                vecs = self._mean_pool(out.last_hidden_state,
                                       encoded["attention_mask"])
                vecs = F.normalize(vecs, p=2, dim=1)
                all_vecs.append(vecs.cpu().float().numpy())
        else:
            for start in range(0, len(texts), batch_size):
                batch   = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    out  = self.model(**encoded)
                vecs = self._mean_pool(out.last_hidden_state,
                                       encoded["attention_mask"])
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
    speed_est = 3500 if torch.cuda.is_available() else 80
    est_h     = max(0, total - skip_lines) / speed_est / 3600

    print(f"Chunks total    : {n_chunks}")
    print(f"Already done    : {len(done_chunks)}")
    print(f"To encode now   : {n_chunks - len(done_chunks)}")
    print(f"Est. time       : ~{est_h:.1f}h  "
          f"({'GPU ~3500 art/s' if torch.cuda.is_available() else 'CPU'})\n")

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
                                 chunk_idx, n_chunks, total, all_pmids, t_global)
                mark_chunk_done(chunk_idx)
                pbar.update(len(text_buffer))
                chunk_idx  += 1
                text_buffer = []
                pmid_buffer = []

        if text_buffer:
            _encode_and_save(encoder, text_buffer, pmid_buffer,
                             chunk_idx, n_chunks, total, all_pmids, t_global)
            mark_chunk_done(chunk_idx)
            pbar.update(len(text_buffer))

    print(f"\n[OK] Encoding complete — {len(all_pmids):,} articles.\n")
    return all_pmids


def _encode_and_save(encoder, texts, pmids, chunk_idx,
                     n_chunks, total, all_pmids_out, t_global):
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

    print(f"   Shape  : {dense_vecs.shape}  →  "
          f"{os.path.getsize(dense_file)/1024**2:.0f} MB")
    print(f"   Time   : {elapsed:.0f}s  |  Speed: {speed:.0f} art/s  |  ETA: ~{eta_h:.1f}h")


# ────────────────────────────────────────────────────────────────── #
#  STEP 2 — BUILD FAISS INDEX                                        #
# ────────────────────────────────────────────────────────────────── #

def build_faiss_index(all_pmids: list):
    import faiss, torch

    chunk_files   = sorted(Path(CHUNK_DIR).glob("chunk_*_dense.npy"))
    use_gpu_faiss = (torch.cuda.is_available() and
                     hasattr(faiss, "StandardGpuResources"))

    print("=" * 60)
    print(f"Building FAISS IVFFlat index")
    print(f"  Vectors : {len(all_pmids):,}  DIM={DIM}  NLIST={NLIST}")
    print(f"  FAISS   : {'GPU' if use_gpu_faiss else 'CPU'}")
    print("=" * 60 + "\n")

    if not use_gpu_faiss:
        faiss.omp_set_num_threads(os.cpu_count())

    n_train = min(3, len(chunk_files))
    print(f"Loading {n_train} chunk(s) for IVF training...")
    sample  = np.vstack([np.load(cf) for cf in chunk_files[:n_train]]).astype("float32")
    print(f"  {len(sample):,} vectors\n")

    quantizer = faiss.IndexFlatIP(DIM)
    index_cpu = faiss.IndexIVFFlat(quantizer, DIM, NLIST, faiss.METRIC_INNER_PRODUCT)

    if use_gpu_faiss:
        res       = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        print("Training on GPU...")
        t0 = time.time()
        index_gpu.train(sample); del sample
        print(f"[OK] {(time.time()-t0)/60:.1f} min\n")
        index_gpu.nprobe = NPROBE
        print("Adding vectors...")
        t0 = time.time()
        for cf in tqdm(chunk_files, desc="Adding", unit="chunk"):
            vecs = np.load(cf).astype("float32")
            index_gpu.add(vecs); del vecs
        print(f"[OK] {index_gpu.ntotal:,} vectors in {(time.time()-t0)/60:.1f} min\n")
        index_final = faiss.index_gpu_to_cpu(index_gpu)
    else:
        print("Training on CPU (~10-30 min)...")
        t0 = time.time()
        index_cpu.train(sample); del sample
        print(f"[OK] {(time.time()-t0)/60:.1f} min\n")
        index_cpu.nprobe = NPROBE
        print("Adding vectors...")
        t0 = time.time()
        for cf in tqdm(chunk_files, desc="Adding", unit="chunk"):
            vecs = np.load(cf).astype("float32")
            index_cpu.add(vecs); del vecs
        print(f"[OK] {index_cpu.ntotal:,} vectors in {(time.time()-t0)/60:.1f} min\n")
        index_final = index_cpu

    print(f"Saving FAISS index → {FAISS_FILE}  (~38GB, few minutes)...")
    t0 = time.time()
    faiss.write_index(index_final, FAISS_FILE)
    print(f"[OK] {os.path.getsize(FAISS_FILE)/1024**3:.1f} GB in {(time.time()-t0)/60:.1f} min\n")

    with open(PMID_MAP, "w") as f:
        json.dump(all_pmids, f)
    print(f"[OK] {len(all_pmids):,} PMIDs saved → {PMID_MAP}\n")


# ────────────────────────────────────────────────────────────────── #
#  STEP 3 — SANITY CHECK                                             #
# ────────────────────────────────────────────────────────────────── #

def sanity_check():
    import faiss
    print("=" * 60)
    print("Sanity check — 3 biomedical test queries")
    print("=" * 60 + "\n")

    index = faiss.read_index(FAISS_FILE)
    index.nprobe = NPROBE
    with open(PMID_MAP) as f:
        pmid_list = json.load(f)

    from huggingface_hub import snapshot_download
    encoder = BiomedBERTEncoder(snapshot_download(repo_id=MODEL_NAME))

    for query in [
        "diabetes insulin resistance treatment metformin",
        "COVID-19 antiviral therapy remdesivir",
        "BRCA1 mutation breast cancer risk hereditary",
    ]:
        vec         = encoder.encode([query], batch_size=1)
        dists, idxs = index.search(vec, 5)
        print(f"  Query: '{query}'")
        for rank, (idx, dist) in enumerate(zip(idxs[0], dists[0]), 1):
            if idx != -1:
                print(f"    [{rank}] PMID={pmid_list[idx]}  score={dist:.4f}")
        print()

    print("[OK] Sanity check passed!\n")
    print(f"  Copy to local machine:")
    print(f"    {FAISS_FILE}")
    print(f"    {PMID_MAP}")
    print(r"  Destination: C:\projects\BioASQ13B\data\indexes\dense_biomedbert\\")


# ────────────────────────────────────────────────────────────────── #
#  MAIN                                                              #
# ────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 60)
    print("  DENSE INDEX BUILDER — BiomedBERT-large + FAISS")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Input  : {JSONL_FILE}")
    print(f"  Output : {DENSE_DIR}")
    print(f"  Chunk  : {CHUNK_SIZE:,}  Batch: {BATCH_SIZE}  MaxLen: {MAX_LENGTH}")
    print(f"  DataLoader workers: {NUM_WORKERS}  compile: {USE_TORCH_COMPILE}")
    print("=" * 60 + "\n")

    check_deps()
    check_disk()

    os.makedirs(DENSE_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    if not os.path.exists(JSONL_FILE):
        print(f"[STOP] JSONL not found: {JSONL_FILE}")
        sys.exit(1)

    done_chunks = load_progress()

    # ── Article count — cached after first run ────────────────────
    total    = get_total_articles()
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

    # ── STEP 2: FAISS ─────────────────────────────────────────────
    if os.path.exists(FAISS_FILE):
        print(f"[INFO] FAISS index exists "
              f"({os.path.getsize(FAISS_FILE)/1024**3:.1f} GB) — skipping.\n")
    else:
        build_faiss_index(all_pmids)

    # ── STEP 3: Sanity check ──────────────────────────────────────
    sanity_check()

    print("=" * 60)
    print("  ALL DONE")
    print(f"  {FAISS_FILE}")
    print(f"  {PMID_MAP}")
    print("=" * 60)


if __name__ == "__main__":
    main()