"""
preprocessing.py
-----------------
Text cleaning and NLP feature extraction utilities for the resume
screening system.

Two layers of NLP are used:
1. Lexicon/regex-based skill extraction (skill_taxonomy.py) run on lightly
   cleaned text so multi-word skill phrases ("machine learning",
   "power bi") survive intact.
2. Classic NLTK preprocessing (tokenize -> stopword removal -> lemmatize)
   to produce a clean bag-of-words representation used for TF-IDF
   similarity scoring between a resume and a job description.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from skill_taxonomy import FLAT_SKILLS, ALL_SKILL_NAMES

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Precompile skill regex patterns once (case-insensitive, word-boundary safe)
_COMPILED_SKILLS = []
for canonical, meta in FLAT_SKILLS.items():
    for pat in meta["patterns"]:
        _COMPILED_SKILLS.append((canonical, meta["category"], re.compile(pat, re.IGNORECASE)))


def light_clean(text: str) -> str:
    """Light cleaning that preserves punctuation-sensitive tokens like C++, .NET, C#."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> list:
    """Return sorted list of canonical skill names found in the text."""
    text = light_clean(text)
    found = set()
    for canonical, category, pattern in _COMPILED_SKILLS:
        if pattern.search(text):
            found.add(canonical)
    return sorted(found)


def deep_clean_for_tfidf(text: str) -> str:
    """Full NLTK pipeline: lowercase -> tokenize -> remove stopwords/punct -> lemmatize."""
    text = light_clean(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s]", " ", text)  # keep tokens like c++, .net, c#
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        if tok in _STOPWORDS or len(tok) <= 1:
            continue
        if tok.isdigit():
            continue
        lemma = _LEMMATIZER.lemmatize(tok, pos="v")
        lemma = _LEMMATIZER.lemmatize(lemma, pos="n")
        cleaned.append(lemma)
    return " ".join(cleaned)


def skill_gap(resume_skills: list, required_skills: list) -> dict:
    """Compare resume skills against a job's required skills."""
    r = set(resume_skills)
    j = set(required_skills)
    matched = sorted(r & j)
    missing = sorted(j - r)
    extra = sorted(r - j)
    coverage = len(matched) / len(j) if j else 0.0
    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "coverage": round(coverage, 4),
    }
