# BioASQ 13b: A Multi-Stage Pipeline for Biomedical Question Answering

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/transformers/)
[![PyTorch](https.img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pyserini (BM25)](https://img.shields.io/badge/Pyserini-BM25-orange)](https://github.com/castorini/pyserini)

This repository contains the code for our submission to the **BioASQ Challenge 13, Task B**. Our system is a multi-stage pipeline that first retrieves relevant documents and then uses them to generate precise answers.

**Our Core Approach:**
1.  **Phase A (Retrieval):** We use a hybrid retrieval approach. An initial set of candidate documents is fetched using a traditional sparse retriever (BM25). These candidates are then re-ranked using a fine-tuned BERT-based cross-encoder to improve relevance.
2.  **Phase B (Generation):** The top-ranked documents from Phase A are fed as context to a generative model to produce the final factoid, list, or summary answers.

---

## System Architecture

Our pipeline processes a question in sequential phases to arrive at the final answer.

![System Architecture Diagram]([path/to/your/diagram.png])  <!-- It's highly recommended to create a simple diagram in a tool like draw.io and add it here -->

**1. BM25 Indexing & Search (`phaseA-BM25`):**
   - A searchable index of the biomedical literature is created.
   - For an incoming question, this module performs a fast, keyword-based search to retrieve a large set of potentially relevant documents (e.g., top 100).

**2. Neural Reranking (`phaseA-reranker`):**
   - The documents from the BM25 search are passed to a fine-tuned cross-encoder model (e.g., BioBERT).
   - This model scores each `(question, document)` pair for relevance, producing a more accurate ranking.

**3. Answer Generation (`phaseB`, `phaseAp`):**
   - The top N most relevant documents (e.g., top 5) from the reranker are concatenated to form a context.
   - The question and the context are passed to a language model to generate the final answer in the required format.

---

## Performance

<!-- Fill this in with your best scores on the development or test set -->
Our model achieves the following results on the official BioASQ 13b development set:

| Question Type | Metric      | Score      |
|---------------|-------------|------------|
| Factoid       | Accuracy    | **[0.XX]** |
|               | MRR         | **[0.XX]** |
| List          | F1-Score    | **[0.XX]** |
| Yes/No        | Accuracy    | **[0.XX]** |

---

## Setup and Installation

Follow these steps to set up the environment and prepare the necessary data and models.

### 1. Prerequisites
*   Python 3.9+
*   A system with sufficient RAM and a modern NVIDIA GPU (for the reranker and generation phases).
*   ~[XX] GB of disk space.

### 2. Clone the Repository
```sh
git clone [your-github-repo-link]
cd [your-repo-name]
```

### 3. Install Dependencies
Create and activate a virtual environment, then install the required packages.
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download Data & Build Indexes
You must download the official BioASQ datasets and build the BM25 index.
```sh
# Download the baseline data (update script if necessary)
python data/baselines/download_baselines.py

# Create the BM25 search index
python phaseA-BM25/create_indexes.py --path [path/to/bioasq/corpus]

# Download our fine-tuned models (if you're hosting them)
# [Add instructions here, e.g., using wget, git-lfs, or manual download from a drive]
```

---

## Running the Pipeline

The easiest way to run the full pipeline is by using the provided shell scripts in the `/scripts/Sample` directory. **Please inspect these scripts and update any hardcoded paths before running.**

### Phase A: Document Retrieval & Reranking

This phase trains the reranker and then uses it to process a set of questions.

```sh
cd scripts/Sample/phaseA/

# 1. Train the reranker model (if not using a pre-trained one)
bash 1_trainer.sh

# 2. Rerank the documents for a given test file
bash 2_reranker.sh

# 3-6. Convert outputs to the required formats for evaluation/next steps
bash 3_convert.sh
# ... and so on for the other scripts.
```

### Phase B: Answer Generation

This phase takes the reranked documents and generates the final answers.

```sh
cd scripts/Sample/phaseB/  # or phaseAp

# 1. Look up abstracts for the top documents
bash 1_abstract_lookup.sh

# 2. Generate initial answers using an LLM or custom model
bash 2_initial_gen.sh

# 3. Post-process into final summaries/answers
bash 3_summaries.sh

# 4. Convert to the official BioASQ submission format
bash 4_convert.sh
```

---

## Directory Structure

A brief overview of the key directories in this project.
```
├── data/                  # Scripts for downloading, processing, and managing data
├── phaseA-BM25/           # BM25 sparse retriever: indexing and searching
├── phaseA-reranker/       # BERT-based cross-encoder: training and inference
├── phaseB/                # Answer generation and summarization logic
├── phaseAp/               # Alternative/experimental generation logic
├── scripts/               # Wrapper scripts to execute the full pipeline
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## License

Distributed under the [MIT License]. See `LICENSE.txt` for more information.

## Contact

[Your Name / Team Name] - [@YourTwitter(optional)] - [your_email@domain.com]

Project Link: [https://github.com/your_username/your_repository](https://github.com/your_username/your_repository)