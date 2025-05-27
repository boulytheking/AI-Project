"""
Prédit IA ou Humain pour un texte donné.
Usage :
python -m src.predict --text "Your essay here..."
ou
python -m src.predict --file my_essay.txt
"""
import argparse, joblib
from pathlib import Path
from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "final_model" / "minilm_svm.joblib"
bundle = joblib.load(MODEL_PATH)
embedder  = bundle["embedder"]
classifier = bundle["classifier"]

parser = argparse.ArgumentParser()
parser.add_argument("--text", help="Texte brut à analyser")
parser.add_argument("--file", help="Chemin d'un fichier texte")
args = parser.parse_args()

if args.file:
    text = Path(args.file).read_text(encoding="utf-8")
else:
    text = args.text or ""

vec   = embedder.encode([clean_text(text)])
pred  = classifier.predict(vec)[0]
print("Humain" if pred == 0 else "Texte généré par IA")
