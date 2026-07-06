"""
generate_resumes.py
--------------------
Generates a simulated resume dataset (as permitted by the task brief:
"simulated resumes... as long as the data reflects real hiring
scenarios, it is valid"). No real resume dataset was supplied, so this
creates ~130 realistic, structured candidate resumes spanning the same
seven IT role families found in the real Monster.com job-postings pool,
with deliberately varied skill completeness, seniority, and education so
that ranking / screening produces a meaningful, realistic spread.
"""
import random
import pandas as pd

random.seed(42)

OUT_PATH = "/home/claude/resume_screening/data/synthetic_resumes.csv"

FIRST_NAMES = [
    "Amara", "Liam", "Sipho", "Chen", "Fatima", "Noah", "Thandiwe", "Mateo",
    "Priya", "Ethan", "Zanele", "Lucas", "Aisha", "James", "Naledi", "Daniel",
    "Yara", "Michael", "Kagiso", "Sarah", "Omar", "Emily", "Bongani", "Ryan",
    "Lerato", "David", "Nadia", "Kevin", "Palesa", "Andrew", "Mei", "Jordan",
    "Karabo", "Rachel", "Tumelo", "Grace", "Ahmed", "Olivia", "Sibusiso", "Anna",
]
LAST_NAMES = [
    "Nkosi", "Smith", "Dlamini", "Chen", "Khumalo", "Johnson", "Naidoo", "Williams",
    "Mokoena", "Brown", "Patel", "Davies", "Sithole", "Miller", "Van der Merwe",
    "Wilson", "Mahlangu", "Taylor", "Govender", "Anderson", "Zulu", "Thomas",
    "Ndlovu", "Moore", "Reddy", "Jackson", "Botha", "White", "Mabaso", "Harris",
]

EDUCATION_LEVELS = ["Diploma", "Bachelor's Degree", "Bachelor's Degree", "Honours Degree", "Master's Degree"]
DEGREE_FIELDS = {
    "IT Support": ["Information Technology", "Computer Science", "IT Support & Networking"],
    "Cyber Security": ["Cyber Security", "Computer Science", "Information Systems"],
    "Networks & Systems": ["Information Technology", "Computer Engineering", "Network Engineering"],
    "Software Development": ["Computer Science", "Software Engineering", "Information Systems"],
    "Data & Business Analytics": ["Mathematical Statistics", "Data Science", "Computer Science", "Business Analytics"],
    "QA & Testing": ["Computer Science", "Information Technology", "Software Engineering"],
    "IT Project Management": ["Business Information Systems", "Project Management", "Information Systems"],
}

CERTIFICATIONS = {
    "IT Support": ["CompTIA A+", "ITIL Foundation", "HDI Support Technician"],
    "Cyber Security": ["CompTIA Security+", "CEH (Certified Ethical Hacker)", "CISSP Associate"],
    "Networks & Systems": ["CCNA", "CompTIA Network+", "Microsoft Certified: Azure Administrator"],
    "Software Development": ["AWS Certified Developer", "Oracle Certified Java Programmer", None],
    "Data & Business Analytics": ["Microsoft Power BI Data Analyst", "Google Data Analytics Certificate", None],
    "QA & Testing": ["ISTQB Foundation", "Certified Selenium Tester", None],
    "IT Project Management": ["PMP", "PRINCE2 Foundation", "Certified ScrumMaster"],
}

# Core skill pools per target role family (drawn from the shared taxonomy)
CORE_SKILLS = {
    "IT Support": ["Call Tracking Software", "Ticketing Systems", "Hardware Troubleshooting",
                   "Software Troubleshooting", "Customer Support", "Microsoft Office Suite",
                   "Phone-based Support", "Remote Desktop Tools", "Active Directory", "Windows Server"],
    "Cyber Security": ["Cyber Security", "Penetration Testing", "Vulnerability Assessment", "SIEM",
                        "Risk Assessment", "Compliance (SOX/HIPAA/PCI)", "Incident Response", "Firewalls", "VPN"],
    "Networks & Systems": ["Active Directory", "Windows Server", "Networking (LAN/WAN)", "DNS/DHCP",
                            "Firewalls", "VPN", "System Administration", "Data Center Operations",
                            "Linux", "Virtualization", "AWS", "Azure"],
    "Software Development": ["Python", "Java", "JavaScript", "C#", "SQL", ".NET", "React", "Node.js",
                              "REST APIs", "Git", "Docker", "HTML/CSS", "Microservices"],
    "Data & Business Analytics": ["SQL", "Python", "Excel", "Power BI", "Tableau", "Power Query",
                                   "Pandas", "Data Visualization", "Statistical Analysis",
                                   "ETL", "Data Warehousing"],
    "QA & Testing": ["Manual Testing", "Automated Testing", "Selenium", "Test Case Design",
                      "Regression Testing", "Defect Tracking", "Load/Performance Testing", "Agile", "SQL"],
    "IT Project Management": ["Project Management", "Agile", "Scrum", "Stakeholder Management",
                               "Requirements Gathering", "Budgeting", "PMP", "Kanban", "Six Sigma"],
}

SOFT_SKILLS_POOL = ["Communication", "Problem Solving", "Teamwork", "Leadership",
                     "Time Management", "Documentation"]

CROSSOVER_POOL = ["Git", "Microsoft Office Suite", "SQL", "Agile", "Communication",
                   "Customer Support", "Excel", "Linux"]

SUMMARY_TEMPLATES = [
    "{level} {role} with {years} years of experience supporting {domain}. "
    "Known for {trait1} and {trait2}, with a track record of {impact}.",
    "{level} professional specializing in {role_lower}, bringing {years} years of hands-on "
    "experience in {domain}. Combines {trait1} with {trait2} to deliver {impact}.",
    "Results-driven {role_lower} with {years} years in the field. Strong background in {domain}, "
    "recognized for {trait1}. Passionate about {impact}.",
]
TRAITS = ["clear communication", "fast troubleshooting", "attention to detail", "cross-team collaboration",
          "process improvement", "analytical rigor", "stakeholder management", "mentoring junior staff"]
DOMAINS = {
    "IT Support": "enterprise help desk and end-user support operations",
    "Cyber Security": "security operations, risk management, and incident response",
    "Networks & Systems": "network infrastructure and systems administration",
    "Software Development": "full-stack application development",
    "Data & Business Analytics": "data analysis, reporting, and business intelligence",
    "QA & Testing": "software quality assurance and test automation",
    "IT Project Management": "IT project delivery and agile program management",
}
IMPACTS = ["reducing resolution time", "improving system reliability", "driving data-informed decisions",
           "shipping features on schedule", "cutting production defects", "keeping stakeholders aligned"]


def make_name(used):
    while True:
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name


def seniority(years):
    if years <= 1:
        return "Junior"
    if years <= 4:
        return "Mid-level"
    return "Senior"


def build_skill_set(role_family, strength):
    """strength in {'strong','average','weak'} controls how many core skills the candidate has."""
    core = CORE_SKILLS[role_family][:]
    random.shuffle(core)
    if strength == "strong":
        n_core = random.randint(max(6, len(core) - 2), len(core))
    elif strength == "average":
        n_core = random.randint(4, max(5, len(core) - 3))
    else:
        n_core = random.randint(1, 3)
    skills = set(core[:n_core])

    # sprinkle a couple of crossover / soft skills so resumes read naturally
    skills.update(random.sample(CROSSOVER_POOL, k=random.randint(1, 2)))
    skills.update(random.sample(SOFT_SKILLS_POOL, k=random.randint(1, 2)))

    # small chance of a stray skill from a totally different family (career changers)
    if random.random() < 0.15:
        other_family = random.choice([f for f in CORE_SKILLS if f != role_family])
        skills.add(random.choice(CORE_SKILLS[other_family]))

    return sorted(skills)


def build_resume_text(name, role_family, years, level, education, degree_field, cert, skills):
    domain = DOMAINS[role_family]
    template = random.choice(SUMMARY_TEMPLATES)
    summary = template.format(
        level=level, role=role_family, role_lower=role_family.lower(), years=years,
        domain=domain, trait1=random.choice(TRAITS), trait2=random.choice(TRAITS),
        impact=random.choice(IMPACTS),
    )
    skills_line = "Skills: " + ", ".join(skills) + "."
    edu_line = f"Education: {education} in {degree_field}."
    cert_line = f"Certifications: {cert}." if cert else ""
    exp_line = (f"Experience: {years} year(s) in {role_family} roles, most recently as a "
                f"{level} {role_family} professional.")
    parts = [f"{name} - {level} {role_family}", summary, exp_line, skills_line, edu_line, cert_line]
    return " ".join(p for p in parts if p)


def generate(n_per_family=19):
    used_names = set()
    rows = []
    cid = 1
    for role_family in CORE_SKILLS:
        for i in range(n_per_family):
            strength = random.choices(["strong", "average", "weak"], weights=[0.35, 0.4, 0.25])[0]
            years = random.choice([0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12])
            level = seniority(years)
            education = random.choice(EDUCATION_LEVELS)
            degree_field = random.choice(DEGREE_FIELDS[role_family])
            cert = random.choice(CERTIFICATIONS[role_family])
            skills = build_skill_set(role_family, strength)
            name = make_name(used_names)
            resume_text = build_resume_text(name, role_family, years, level, education,
                                             degree_field, cert, skills)
            rows.append({
                "candidate_id": "C" + str(cid).zfill(4),
                "name": name,
                "target_role_family": role_family,
                "years_experience": years,
                "seniority": level,
                "education_level": education,
                "degree_field": degree_field,
                "certification": cert or "",
                "skill_strength": strength,
                "skills": "; ".join(skills),
                "resume_text": resume_text,
            })
            cid += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH, "rows:", len(df))
    print(df["target_role_family"].value_counts())
    print(df["skill_strength"].value_counts())
