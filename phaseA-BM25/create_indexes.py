import os
import sys
import time
import subprocess

# ══════════════════════════════════════════════════════════════════ #
#  PYSERINI INDEXING SCRIPT                                         #
#  Works for both test runs and full 38M article runs              #
#                                                                  #
#  Before running:                                                  #
#      pip install pyserini                                        #
#                                                                  #
#  To run test (6 million articles):                               #
#      python create_index.py --mode test                          #
#                                                                  #
#  To run full index (all 38 million articles):                    #
#      python create_index.py --mode full                          #
# ══════════════════════════════════════════════════════════════════ #


# ------------------------------------------------------------------ #
#  CONFIGURE — change these paths to match your setup               #
# ------------------------------------------------------------------ #

BASE_DIR   = r"C:\projects\BioASQ13B"

FULL_JSONL = os.path.join(BASE_DIR, "data", "full_input", "pubmed_collection.jsonl")
TEST_JSONL = os.path.join(BASE_DIR, "data", "test_input", "pubmed_test_6m.jsonl")
#TEST_INDEX = os.path.join(BASE_DIR, "data", "indexes",    "pyserini_test")
FULL_INDEX = os.path.join(BASE_DIR, "data", "indexes",    "pyserini_pubmed_full")

TEST_ARTICLE_LIMIT = 6_000_000

# 2 = safe for 8GB RAM  |  4 = faster but needs 16GB+
THREADS = 2

# ------------------------------------------------------------------ #


# ══════════════════════════════════════════════════════════════════ #
#  HELPERS                                                          #
# ══════════════════════════════════════════════════════════════════ #

def check_pyserini():
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa
        print("[OK] Pyserini is installed.\n")
    except ImportError:
        print("[STOP] Pyserini is not installed.")
        print("       Run:  pip install pyserini")
        sys.exit(1)


def check_disk_space(path, needed_gb):
    import shutil
    check_path = path
    while check_path and not os.path.exists(check_path):
        parent = os.path.dirname(check_path)
        if parent == check_path:
            check_path = os.path.splitdrive(path)[0] + os.sep
            break
        check_path = parent
    try:
        free_gb = shutil.disk_usage(check_path).free / (1024 ** 3)
        print(f"[INFO] Free disk space : {free_gb:.1f} GB")
        print(f"[INFO] Estimated need  : {needed_gb:.1f} GB")
        if free_gb < needed_gb:
            print(f"[WARNING] Low disk space — free up {needed_gb:.0f} GB before indexing.\n")
        else:
            print(f"[OK] Enough disk space available.\n")
    except Exception as e:
        print(f"[INFO] Could not check disk space: {e} — continuing.\n")


def check_input_folder(jsonl_path):
    input_dir  = os.path.dirname(jsonl_path)
    jsonl_name = os.path.basename(jsonl_path)

    if not os.path.exists(input_dir):
        print(f"[STOP] Input folder not found: {input_dir}")
        sys.exit(1)

    if not os.path.exists(jsonl_path):
        print(f"[STOP] JSONL file not found: {jsonl_path}")
        sys.exit(1)

    all_files   = [f for f in os.listdir(input_dir) if not f.startswith(".")]
    other_files = [f for f in all_files if f != jsonl_name]

    if other_files:
        print(f"[WARNING] Input folder contains {len(other_files)} unexpected file(s):")
        for f in other_files[:5]:
            print(f"    {f}")
        if len(other_files) > 5:
            print(f"    ... and {len(other_files) - 5} more")
        print("\n  Pyserini indexes ALL files in the folder.")
        print("  Move these files out to avoid indexing wrong data.\n")
        answer = input("  Continue anyway? [y/N]: ").strip().lower()
        if answer != "y":
            print("\nAborted. Clean up the folder and re-run.")
            sys.exit(0)
        print()
    else:
        print(f"[OK] Input folder is clean — only 1 file found: {jsonl_name}\n")


# ══════════════════════════════════════════════════════════════════ #
#  STEP 1 — CREATE TEST JSONL                                       #
# ══════════════════════════════════════════════════════════════════ #

def create_test_jsonl(limit):
    if not os.path.exists(FULL_JSONL):
        print(f"[STOP] Full JSONL not found at: {FULL_JSONL}")
        sys.exit(1)

    os.makedirs(os.path.dirname(TEST_JSONL), exist_ok=True)

    if os.path.exists(TEST_JSONL):
        print(f"[INFO] Test JSONL already exists. Counting lines ...")
        with open(TEST_JSONL, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= limit:
            print(f"[OK] Test JSONL has {existing:,} articles — skipping creation.\n")
            return
        else:
            print(f"[INFO] Only {existing:,} articles found — recreating with {limit:,} ...\n")

    print(f"Creating test JSONL with first {limit:,} articles ...")
    t0    = time.time()
    count = 0

    with open(FULL_JSONL, "r", encoding="utf-8") as fin, \
         open(TEST_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            if count >= limit:
                break
            fout.write(line)
            count += 1
            if count % 500_000 == 0:
                print(f"  {count:>10,} articles written  ({time.time() - t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    size_gb = os.path.getsize(TEST_JSONL) / (1024 ** 3)
    print(f"\n[OK] Test JSONL created — {count:,} articles  |  {size_gb:.1f} GB  |  {elapsed:.0f}s\n")


# ══════════════════════════════════════════════════════════════════ #
#  STEP 2 — BUILD PYSERINI INDEX                                    #
# ══════════════════════════════════════════════════════════════════ #

def build_index(input_jsonl, index_path, label):
    print("=" * 55)
    print(f"Building {label} index ...")
    print(f"  Input  : {input_jsonl}")
    print(f"  Index  : {index_path}")
    print(f"  Threads: {THREADS}")
    print("=" * 55 + "\n")

    os.makedirs(index_path, exist_ok=True)

    segments = [f for f in os.listdir(index_path)
                if f.startswith("segment") or f.endswith(".si")]
    if segments:
        print(f"[INFO] Index already has {len(segments)} segment(s) — skipping build.")
        print(f"       To rebuild, manually delete: {index_path}\n")
        return True

    check_input_folder(input_jsonl)

    t0        = time.time()
    input_dir = os.path.dirname(input_jsonl)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection",  "JsonCollection",
        "--input",        input_dir,
        "--index",        index_path,
        "--generator",   "DefaultLuceneDocumentGenerator",
        "--threads",      str(THREADS),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print("Running Pyserini indexer — do NOT close this terminal.\n")
    result  = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] Indexer exited with code {result.returncode}")
        return False

    print(f"\n[OK] Indexing complete in {elapsed / 60:.1f} minutes.")
    return True


# ══════════════════════════════════════════════════════════════════ #
#  MAIN                                                             #
# ══════════════════════════════════════════════════════════════════ #

def main():
    mode = "full"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().strip("-")
        if arg in ("full", "f"):
            mode = "full"
        elif arg in ("test", "t"):
            mode = "test"
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage:")
            print("  python create_index.py --mode test   (6M articles)")
            print("  python create_index.py --mode full   (all 38M articles)")
            sys.exit(1)

    print("=" * 55)
    if mode == "test":
        print(f"  PYSERINI INDEXING — TEST MODE")
        print(f"  Articles : first {TEST_ARTICLE_LIMIT:,}")
    else:
        print(f"  PYSERINI INDEXING — FULL MODE")
        print(f"  Articles : all (~38 million)")
    print("=" * 55 + "\n")

    check_pyserini()

    if mode == "test":
        input_jsonl = TEST_JSONL
        #index_path  = TEST_INDEX
        needed_gb   = 15.0
        label       = "test"
    else:
        input_jsonl = FULL_JSONL
        index_path  = FULL_INDEX
        needed_gb   = 35.0
        label       = "full"

    check_disk_space(index_path, needed_gb)

    if mode == "test":
        create_test_jsonl(TEST_ARTICLE_LIMIT)
    else:
        if not os.path.exists(FULL_JSONL):
            print(f"[STOP] Full JSONL not found at: {FULL_JSONL}")
            sys.exit(1)
        size_gb = os.path.getsize(FULL_JSONL) / (1024 ** 3)
        print(f"[OK] Full JSONL found: {size_gb:.1f} GB\n")

    success = build_index(input_jsonl, index_path, label)

    print("=" * 55)
    if success:
        print(f"  ALL INDEXED DONE")
        print(f"  Index is ready at: {index_path}")
        print()
        print(f"  Next step — run BM25 grid search:")
        print(f"      python grid_search_pyserini.py --mode {mode}")
    else:
        print(f"  INDEXING FAILED — check errors above.")
    print("=" * 55)


if __name__ == "__main__":
    main()