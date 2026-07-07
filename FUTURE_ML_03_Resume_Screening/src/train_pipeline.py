"""
train_pipeline.py
------------------
Runs the full pipeline end-to-end and saves all artifacts needed by the
Streamlit dashboard:
  - artifacts/jobs_processed.pkl      (jobs + clean_text + skills_list + req_years)
  - artifacts/resumes_processed.pkl   (resumes + clean_text + skills_list)
  - artifacts/tfidf_vectorizer.pkl
  - artifacts/fit_model.pkl           (best classifier)
  - artifacts/model_metrics.json
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from preprocessing import deep_clean_for_tfidf, extract_skills
from scoring import (fit_vectorizer, extract_required_years, build_pair_features,
                      sample_training_pairs, composite_score)

DATA_DIR = "/home/claude/resume_screening/data"
ART_DIR = "/home/claude/resume_screening/artifacts"

FEATURE_COLS = ["tfidf_similarity", "skill_coverage", "n_matched_skills", "n_missing_skills",
                 "n_extra_skills", "years_experience", "required_years", "experience_gap",
                 "role_family_match", "education_score"]


def load_and_preprocess():
    jobs = pd.read_csv(f"{DATA_DIR}/job_postings.csv")
    resumes = pd.read_csv(f"{DATA_DIR}/synthetic_resumes.csv")

    jobs["clean_text"] = jobs["job_description"].apply(deep_clean_for_tfidf)
    jobs["skills_list"] = jobs["job_description"].apply(extract_skills)
    jobs["n_skills"] = jobs["skills_list"].apply(len)
    jobs = jobs[jobs["n_skills"] >= 2].reset_index(drop=True)  # need identifiable requirements
    jobs["req_years"] = jobs["job_description"].apply(extract_required_years)

    resumes["clean_text"] = resumes["resume_text"].apply(deep_clean_for_tfidf)
    resumes["skills_list"] = resumes["skills"].apply(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])

    return jobs, resumes


def main():
    print("Loading & preprocessing...")
    jobs, resumes = load_and_preprocess()
    print(f"  jobs: {len(jobs)}  resumes: {len(resumes)}")

    print("Fitting TF-IDF vectorizer...")
    corpus = pd.concat([jobs["clean_text"], resumes["clean_text"]]).tolist()
    vec = fit_vectorizer(corpus)

    print("Sampling training pairs...")
    pairs = sample_training_pairs(resumes, jobs, vec, n_samples=5000, seed=42)
    print(f"  pairs: {len(pairs)}  positive rate: {pairs['good_fit'].mean():.3f}")

    X = pairs[FEATURE_COLS].copy()
    X["required_years"] = X["required_years"].replace(-1, X["required_years"].median())
    y = pairs["good_fit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42,
                                                 class_weight="balanced"),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "F1": round(f1_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
        }
        fitted[name] = model
        print(f"  {name}: {results[name]}")

    best_name = max(results, key=lambda k: results[k]["F1"])
    best_model = fitted[best_name]
    print(f"Best model: {best_name}")

    # Precompute composite + ML scores are done live in the app; here we just persist artifacts.
    jobs.to_pickle(f"{ART_DIR}/jobs_processed.pkl")
    resumes.to_pickle(f"{ART_DIR}/resumes_processed.pkl")
    with open(f"{ART_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(f"{ART_DIR}/fit_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler if best_name == "LogisticRegression" else None,
                     "feature_cols": FEATURE_COLS}, f)
    with open(f"{ART_DIR}/model_metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name,
                    "n_training_pairs": len(pairs), "positive_rate": round(float(pairs["good_fit"].mean()), 4)}, f, indent=2)

    print("Artifacts saved to", ART_DIR)


if __name__ == "__main__":
    main()
