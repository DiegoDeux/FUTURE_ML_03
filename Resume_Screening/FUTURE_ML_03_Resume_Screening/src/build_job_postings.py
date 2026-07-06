"""
build_job_postings.py
----------------------
Loads the raw Monster.com jobs sample (real-world data), filters it down to
IT / tech-relevant roles, cleans the text, and tags each posting with a
"role_family" so the screening system can compare resumes against a
coherent job category (IT Support, Software Development, Data/Business
Analytics, Networks & Systems, Cyber Security, QA/Testing, IT Project
Management).

Output: data/job_postings.csv
"""
import re
import pandas as pd

RAW_PATH = "/home/claude/resume_screening/data/monster_com-job_sample.csv"
OUT_PATH = "/home/claude/resume_screening/data/job_postings.csv"

# Keyword rules -> role_family (checked in order, first match wins)
ROLE_RULES = [
    ("IT Support", [
        r"help ?desk", r"desktop support", r"it support", r"technical support",
        r"support technician", r"pc technician", r"support analyst", r"support specialist",
        r"service desk",
    ]),
    ("Cyber Security", [
        r"security engineer", r"cyber ?security", r"information security",
        r"security analyst", r"penetration", r"soc analyst",
    ]),
    ("Networks & Systems", [
        r"network engineer", r"systems administrator", r"system administrator",
        r"sys ?admin", r"network administrator", r"data center", r"windows admin",
        r"infrastructure", r"cloud engineer", r"solutions? architect", r"it technician",
    ]),
    ("Software Development", [
        r"software engineer", r"software developer", r"\bdeveloper\b", r"programmer",
        r"full[- ]?stack", r"front[- ]?end", r"back[- ]?end", r"\.net", r"java developer",
        r"web developer", r"application(s)? architect", r"mobile developer", r"devops",
    ]),
    ("Data & Business Analytics", [
        r"data scientist", r"data analyst", r"business analyst", r"data specialist",
        r"database architect", r"database administrator", r"\bdba\b", r"reporting analyst",
        r"bi analyst", r"business intelligence",
    ]),
    ("QA & Testing", [
        r"qa\b", r"quality assurance", r"test engineer", r"testing analyst",
        r"automation engineer", r"quality analyst",
    ]),
    ("IT Project Management", [
        r"project manager", r"program manager", r"scrum master", r"it manager",
        r"delivery manager", r"product manager",
    ]),
]


def categorize(title: str):
    if not isinstance(title, str):
        return None
    t = title.lower()
    for family, patterns in ROLE_RULES:
        for p in patterns:
            if re.search(p, t):
                return family
    return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_location(loc: str) -> str:
    if not isinstance(loc, str):
        return "Not specified"
    loc = loc.strip()
    # Some rows have garbage location values (contact names, repeated titles, etc.)
    if len(loc) > 40 or re.search(r"contact|resume|apply|@", loc, re.I):
        return "Not specified"
    return loc


def main():
    df = pd.read_csv(RAW_PATH, on_bad_lines="skip", engine="python")
    it = df[df["sector"] == "IT/Software Development"].copy()

    it["role_family"] = it["job_title"].apply(categorize)
    it = it.dropna(subset=["role_family", "job_description", "job_title"])

    it["job_description"] = it["job_description"].apply(clean_text)
    it["job_title"] = it["job_title"].apply(lambda t: re.sub(r"\s*Job in\s+.*$", "", str(t)).strip())
    it["location"] = it["location"].apply(clean_location)
    it["organization"] = it["organization"].fillna("Company undisclosed")

    # Require a reasonably substantial description (drop stubs / near-empty rows)
    it = it[it["job_description"].str.len() >= 300]

    # De-duplicate near-identical postings (same title + first 200 chars of description)
    it["dedupe_key"] = it["job_title"].str.lower() + it["job_description"].str[:200]
    it = it.drop_duplicates(subset="dedupe_key")

    keep_cols = ["uniq_id", "job_title", "role_family", "organization", "location",
                 "job_type", "salary", "job_description"]
    it = it[keep_cols].reset_index(drop=True)
    it.insert(0, "job_id", ["J" + str(i + 1).zfill(4) for i in range(len(it))])
    it = it.drop(columns=["uniq_id"])

    it.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH, "rows:", len(it))
    print(it["role_family"].value_counts())


if __name__ == "__main__":
    main()
