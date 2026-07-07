"""
scoring.py
----------
Feature engineering and scoring logic that powers both the training
notebook and the live dashboard:

1. TF-IDF vectorization + cosine similarity between resume and job text
   (semantic/textual match).
2. Lexicon-based skill coverage (matched / missing / extra skills).
3. Experience & education fit signals.
4. A transparent, explainable "Composite Score" (weighted rule-based
   blend recruiters can audit).
5. A synthetic-but-principled labelling scheme + feature matrix used to
   train a supervised "ML Fit Score" classifier, mirroring how a real
   applicant-tracking system would eventually learn from recruiter
   decisions.
"""
import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import deep_clean_for_tfidf, extract_skills, skill_gap

EDU_ORDER = {"Diploma": 1, "Bachelor's Degree": 2, "Honours Degree": 3, "Master's Degree": 4}
YEARS_PATTERN = re.compile(r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?", re.IGNORECASE)


def extract_required_years(job_text: str):
    nums = [int(x) for x in YEARS_PATTERN.findall(job_text) if x.isdigit() and int(x) < 25]
    return min(nums) if nums else np.nan


def fit_vectorizer(all_texts):
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1)
    vec.fit(all_texts)
    return vec


def tfidf_similarity(vec, text_a: str, text_b: str) -> float:
    mat = vec.transform([text_a, text_b])
    sim = cosine_similarity(mat[0], mat[1])[0][0]
    return float(sim)


def build_pair_features(resume, job, vec) -> dict:
    """resume, job: pandas Series with precomputed 'clean_text' and 'skills' columns."""
    sim = tfidf_similarity(vec, resume["clean_text"], job["clean_text"])
    gap = skill_gap(resume["skills_list"], job["skills_list"])
    req_years = job.get("req_years", np.nan)
    years_exp = resume["years_experience"]
    if pd.notna(req_years):
        exp_gap = years_exp - req_years
    else:
        exp_gap = 0.0
    role_match = int(resume["target_role_family"] == job["role_family"])
    edu_score = EDU_ORDER.get(resume["education_level"], 2)

    return {
        "tfidf_similarity": round(sim, 4),
        "skill_coverage": gap["coverage"],
        "n_matched_skills": len(gap["matched"]),
        "n_missing_skills": len(gap["missing"]),
        "n_extra_skills": len(gap["extra"]),
        "years_experience": years_exp,
        "required_years": req_years if pd.notna(req_years) else -1,
        "experience_gap": exp_gap,
        "role_family_match": role_match,
        "education_score": edu_score,
        "_matched_skills": gap["matched"],
        "_missing_skills": gap["missing"],
    }


COMPOSITE_WEIGHTS = {"skill_coverage": 0.55, "tfidf_similarity": 0.30, "experience_bonus": 0.15}


def composite_score(features: dict, weights=None) -> float:
    w = weights or COMPOSITE_WEIGHTS
    exp_gap = features["experience_gap"]
    experience_bonus = 1.0 if exp_gap >= 0 else max(0.0, 1.0 + exp_gap / 5.0)
    score = (
        w["skill_coverage"] * features["skill_coverage"]
        + w["tfidf_similarity"] * features["tfidf_similarity"]
        + w["experience_bonus"] * experience_bonus
    )
    return round(min(score, 1.0) * 100, 2)


def true_fit_probability(features: dict) -> float:
    """Ground-truth generative signal used ONLY to synthesize training labels
    (simulates an underlying recruiter decision process to learn from)."""
    exp_gap = features["experience_gap"]
    experience_term = 1.0 if exp_gap >= 0 else max(0.0, 1.0 + exp_gap / 6.0)
    raw = (
        0.5 * features["skill_coverage"]
        + 0.25 * features["tfidf_similarity"]
        + 0.15 * experience_term
        + 0.10 * features["role_family_match"]
    )
    return raw


FIT_THRESHOLD = 0.28  # calibrated against the observed score distribution (~top ~25-30% = good fit)


def synthesize_label(features: dict, noise=0.05) -> int:
    prob = true_fit_probability(features)
    label = 1 if prob >= FIT_THRESHOLD else 0
    if random.random() < noise:
        label = 1 - label
    return label


def sample_training_pairs(resumes_df, jobs_df, vec, n_samples=4000, seed=42):
    rng = random.Random(seed)
    rows = []
    resume_idx = list(resumes_df.index)
    job_idx = list(jobs_df.index)
    for _ in range(n_samples):
        r = resumes_df.loc[rng.choice(resume_idx)]
        j = jobs_df.loc[rng.choice(job_idx)]
        feats = build_pair_features(r, j, vec)
        label = synthesize_label(feats)
        row = {k: v for k, v in feats.items() if not k.startswith("_")}
        row["candidate_id"] = r["candidate_id"]
        row["job_id"] = j["job_id"]
        row["good_fit"] = label
        rows.append(row)
    return pd.DataFrame(rows)
