# IT Resume / Candidate Screening System
### Future Interns — Machine Learning Task 3 (2026)

An NLP-driven resume screening and ranking system: it reads job postings and resumes, extracts skills,
scores each candidate against a job description, ranks the candidate pool, and highlights skill gaps —
presented through a live "IT Talent Screening Command Center" dashboard.

## What this does

1. **Cleans & categorizes real job postings** from the Monster.com jobs dataset (the exact source
   recommended in the task brief) into 7 IT role families: IT Support, Cyber Security, Networks & Systems,
   Software Development, Data & Business Analytics, QA & Testing, IT Project Management.
2. **Extracts skills** from free text using a curated lexicon of ~90 technical + soft skills across 10
   categories (regex/phrase matching tuned for resumes and job ads).
3. **Scores every resume against a job description** using two blended signals:
   - a transparent, weighted **Composite Score** (skill coverage + TF-IDF semantic similarity + experience fit)
   - a trained **Random Forest classifier** ("ML Fit %") learned from synthetic-but-principled fit labels
4. **Ranks candidates** and shows exactly which required skills each candidate has and is missing.
5. Ships as a **Streamlit dashboard** (`app.py`) a recruiter could actually use.

## Why simulated resumes?

The task brief only supplies a **job descriptions** dataset (Monster.com) — no resume dataset was provided.
The brief explicitly allows this: *"You may also use simulated resumes... as long as the data reflects real
hiring scenarios, it is valid."* 133 realistic, structured resumes were generated
(`src/generate_resumes.py`) across the same 7 role families with deliberately varied seniority, education,
and skill completeness, so ranking produces a genuine, non-trivial spread of candidates.

## Project structure

```
├── app.py                          # Streamlit dashboard (run this!)
├── requirements.txt
├── data/
│   ├── job_postings.csv            # cleaned, categorized real Monster.com IT postings (420 rows)
│   └── synthetic_resumes.csv       # 133 simulated candidate resumes
├── notebooks/
│   └── Resume_Screening_System.ipynb   # full pipeline: EDA -> NLP -> model training -> evaluation
├── src/
│   ├── skill_taxonomy.py           # the skill lexicon
│   ├── preprocessing.py            # NLTK cleaning + skill extraction
│   ├── scoring.py                  # TF-IDF similarity, composite score, training-pair generation
│   ├── build_job_postings.py       # raw Monster.com CSV -> data/job_postings.csv
│   ├── generate_resumes.py         # synthetic resume generator
│   └── train_pipeline.py           # trains the classifier, saves artifacts/
└── artifacts/                      # saved vectorizer + trained model + processed data (used by app.py)
```

## Running it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
streamlit run app.py
```

To regenerate everything from scratch (new synthetic resumes, retrain the model):
```bash
cd src
python generate_resumes.py
python train_pipeline.py
```
(`build_job_postings.py` requires the raw `monster_com-job_sample.csv` — download it from the
[Monster.com jobs dataset on Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom)
and place it in `data/` if you need to rebuild the job-postings pool from the original source.)

## Model performance

Random Forest classifier trained on 5,000 synthetic (resume, job) pairs, 75/25 train/test split:

| Metric | Score |
|---|---|
| Accuracy | 94.8% |
| Precision | 93.1% |
| Recall | 90.0% |
| F1 | 91.5% |
| ROC-AUC | 93.8% |

Feature importance confirms the model learned a sensible decision boundary: **skill coverage** and
**role-family alignment** are the dominant drivers of predicted fit — see the notebook for the full
analysis, confusion matrix, and ROC curve.

## Limitations (disclosed, not hidden)

- The resume pool is simulated — no real resume dataset was supplied for this task.
- Training labels are synthesized from a principled generative rule (not real recruiter decisions), since no
  historical hiring outcome data exists. The same pipeline would retrain directly on real labelled data
  without any architecture changes.

## Tools

Python · NLTK · scikit-learn (TF-IDF, Logistic Regression, Random Forest) · pandas/NumPy · Streamlit · Plotly

---
Built by Diego for the Future Interns Machine Learning internship program.
