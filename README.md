# Local Mortgage Document Intelligence & Compliance RAG Pipeline

An enterprise-ready, privacy-first Retrieval-Augmented Generation (RAG) pipeline built to extract zero-drift financial metrics and dense legal property boundaries from complex, multi-format mortgage packets (e.g., Loan Estimates, Closing Disclosures, and Deeds of Trust). 

The entire system is architected to run completely within a secure local hardware boundary to safeguard sensitive customer financial payloads (SSNs, income verification, property records) with zero external API data leakage.

## 🚀 Key Architectural Features

* **Adaptive Ingestion Layer:** Uses a text-length character heuristic to dynamically route native digital text layers or automatically trigger a high-resolution 300 DPI Otsu-Binarized Tesseract OCR loop for unreadable scanned paperwork.
* **Layout Preservation & Table Serialization:** Custom coordinate tracking loop serializes multi-column tabular data into strict, grid-aligned Markdown strings, completely eliminating horizontal row-drifts and LLM column-swapping hallucinations.
* **Automated Document Classification:** Leverages the local LLM to dynamically classify incoming files, automatically tagging metadata and creating isolated document sub-indices.
* **Isolated Multi-Index Ensemble Retrieval:** Segregates search domains via `LlamaIndex` Semantic Node Splitters and deploys a parallel hybrid retriever combining Dense Vector semantic alignment and Sparse Keyword `BM25` matching (achieving a **100% retrieval hit-rate** during evaluation).
* **Hardware-Aware Memory Guardrails:** Engineered strict `torch.no_grad()` inference gates and aggressive VRAM recycling hooks (`torch.cuda.empty_cache()` / `gc.collect()`) to handle dense, multi-index reasoning queries seamlessly within local GPU memory constraints.

---

## 🛠️ Tech Stack & Frameworks

* **Orchestration & Data Indexing:** `LlamaIndex`
* **Local Model Execution:** `HuggingFace Transformers` / PyTorch (Optimized for Qwen2.5 series local architectures)
* **Keyword Indexing:** `bm25s` (Sparse Search Engine)
* **Computer Vision / OCR:** `Tesseract-OCR`, `OpenCV` (Otsu-Binarization Filters)
* **Document Parsing:** `PyMuPDF`
* **User Interface:** `Gradio` (Modern decoupled state layout)

---

## 📊 Evaluation & System Performance Results

The system was stress-tested against an evaluation grid of complex financial queries and dense legal property matrices:

* **Retrieval Hit Rate:** 100%
* **Retrieval Recall@K:** 93.3%
* **Baseline Digital Response Time:** ~1.5 - 2.4 seconds
* **Peak Complex Reasoning Latency (Local VRAM):** 40 - 50 seconds *(Reflects local ensemble matching under strict non-cloud privacy constraints)*

---

## 💻 Getting Started & Installation

### 1. Prerequisites
Ensure you have system-level Tesseract-OCR dependencies installed on your host machine:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr libtesseract-dev

# macOS
brew install tesseract

# Windows
# Download the installer executable from UB Mannheim and append Tesseract to your Environment System PATH variable
```
2. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Mortgage_Assistant_Chatbot.git](https://github.com/YOUR_USERNAME/Mortgage_Assistant_Chatbot.git)
cd Mortgage_Assistant_Chatbot
```
3. Environment Setup & Installation
Create a local virtual environment and install the required dependencies:

```bash
# Initialize and activate environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
# Install requirements
```
pip install -r requirements.txt
Hardware Note: Running this architecture locally requires a CUDA-compatible GPU (such as an NVIDIA T4 or higher configured with matching CUDA/PyTorch binaries) to successfully accommodate model configurations and manage VRAM recycling flags.
```
📁 Repository Structure
```
Plaintext
├── Chatbot final.ipynb         # Main development notebook & pipeline orchestration
├── requirements.txt            # Core Python package dependencies
└── README.md                   # Project documentation
```

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
