# AI vs Human Essay Detector

Mini-pipeline (Python 3.9)  
- **Embedding :** MiniLM (all-MiniLM-L6-v2)  
- **Classifier :** Linear SVM (class_weight=balanced)  
- **F1 classe IA :** 1.00 sur split 20 %

## Installation rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm fr_core_news_md
pip install sentence-transformers umap-learn gensim
