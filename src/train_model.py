from pathlib import Path
import pandas as pd, joblib
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "combined_essays.csv"
OUT  = ROOT / "outputs" / "final_model"

df = pd.read_csv(DATA)
df["text_pp"] = df["text"].apply(clean_text)

texts = df["text_pp"].tolist()
y      = df["generated"].values

print("Encodage MiniLM…")
model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model_emb.encode(texts, batch_size=128, show_progress_bar=True)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

clf = LinearSVC(class_weight="balanced").fit(X_tr, y_tr)
OUT_PATH = Path(OUT); OUT_PATH.mkdir(parents=True, exist_ok=True)
joblib.dump({"embedder": model_emb, "classifier": clf}, OUT_PATH / "minilm_svm.joblib")
print("Modèle sauvegardé dans", OUT_PATH)
