# Detecting AI-Generated Essays

## 1. Introduction
Context, goal & stakes of AI-text detection.

## 2. Data
- Human essays: Kaggle “AI Text Detection” (1 375)
- AI essays: 50 generated with GPT-Neo-125M  
- Combined: 1 428 texts (3.7 % IA)

## 3. Method
1. **Pre-processing**  
   cleaning, lemmatisation spaCy  
2. **Embeddings compared**  
   TF-IDF, Word2Vec, Doc2Vec, GloVe-100, MiniLM  
3. **Classifier**  
   Logistic Regression / SVM (class_weight = balanced)

## 4. Results
| Embedding | F1 IA |
|-----------|-------|
| TF-IDF    | 0.78 |
| Word2Vec  | 0.90 |
| GloVe-100 | 0.90 |
| Doc2Vec   | 0.55 |
| **MiniLM**| **1.00** |

Include confusion matrix, ROC, UMAP figures.

## 5. Discussion
Why MiniLM wins, limitations (few IA samples), future work (larger corpora, adversarial texts).

## 6. Reproducibility
```bash
pip install -r requirements.txt
python -m src.preprocess --input data/combined_essays.csv --output outputs/full/
python -m src.train_model
python -m src.predict --text "Write your essay…"
