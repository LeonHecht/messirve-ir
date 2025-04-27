# **Train Information Retrieval (IR) models on the messIRve dataset**

🚀 **Fine-tuning IR models using the SentenceTransformerTrainer and custom trainers information retrieval**

---

## **📖 Project Overview**
This project fine-tunes **transformer models** for **retrieving relevant documents** using a contrastive learning approach. It is designed for information retrieval **in the latin american spanish domain** (MessIRve dataset) with **hard negative mining**.

💡 **Key Features**:
- **MultipleNegativesRankingLoss** and **Info-NCE Loss** for contrastive training.
- **Hard negative mining** with `1 positive + 5 hard negatives per query`.
- **Multi-GPU Training** using `DataParallel`.
- **Hyperparameter Sweeps** with `Hydra`.
- **Logging & Visualization** via `Matplotlib` and `Transformers Trainer Logs`.

💡 **Upcoming Features**:
- **Knowledge Distillation**
- **Train Custom Models with Sentence Transformers**

---

## **⚡ Quickstart**
### **1️⃣ Install Dependencies**
```bash
git clone https://github.com/yourusername/messirve-ir.git
cd messirve-ir
pip install -r requirements.txt
