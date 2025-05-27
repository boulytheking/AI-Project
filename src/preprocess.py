# src/preprocess.py
"""
Pré-traitement texte pour la détection IA vs humain.
Usage (CLI) :
python -m src.preprocess --input data/train_essays.csv --output outputs/
"""
from __future__ import annotations
import argparse, html, re
from pathlib import Path

import joblib, pandas as pd, tqdm
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# (optionnel) détection de langue
from langdetect import detect  # pip install langdetect
import spacy

URL_RE = re.compile(r"https?://\S+|www\.\S+")

def clean_text(text) -> str:
    """Nettoie une chaîne. Si text est NaN ou None, renvoie chaîne vide."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = unidecode(text.lower())
    return re.sub(r"\s+", " ", text).strip()

def load_spacy(lang: str):
    try:
        return spacy.load(lang, disable=["ner", "parser"])
    except OSError as e:
        raise RuntimeError(
            f"Modèle spaCy « {lang} » introuvable. "
            "Installe-le avec : python -m spacy download " + lang
        ) from e

NLP_CACHE: dict[str, spacy.language.Language] = {}

def lemmatize(text: str) -> str:
    """Lemmatise en utilisant le meilleur modèle selon la langue détectée."""
    lang_code = "fr" if detect(text[:200]) == "fr" else "en"
    if lang_code not in NLP_CACHE:
        NLP_CACHE[lang_code] = load_spacy("fr_core_news_md" if lang_code == "fr" else "en_core_web_sm")
    doc = NLP_CACHE[lang_code](text)
    return " ".join(
        tok.lemma_ for tok in doc
        if not tok.is_punct and not tok.is_space and not tok.is_stop
    )

def vectorize_and_split(texts: pd.Series, labels: pd.Series, out_dir: Path) -> None:
    vectorizer = TfidfVectorizer(
        max_features=120_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X = vectorizer.fit_transform(texts)
    X_tr, X_val, y_tr, y_val = train_test_split(X, labels, test_size=0.2,
                                                stratify=labels, random_state=42)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")
    joblib.dump(X_tr,      out_dir / "X_train.npz")
    joblib.dump(X_val,     out_dir / "X_val.npz")
    joblib.dump(y_tr,      out_dir / "y_train.pkl")
    joblib.dump(y_val,     out_dir / "y_val.pkl")
    print(f"Artefacts enregistrés dans {out_dir}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,  help="CSV avec colonnes 'text' et 'generated'")
    parser.add_argument("--output", default="outputs/", help="répertoire de sortie")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tqdm.tqdm.pandas()
    df["text_pp"] = df["text"].progress_apply(lambda t: lemmatize(clean_text(t)))
    vectorize_and_split(df["text_pp"], df["generated"], Path(args.output))

if __name__ == "__main__":
    main()
