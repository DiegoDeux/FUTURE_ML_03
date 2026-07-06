"""
skill_taxonomy.py
------------------
A curated taxonomy of IT/tech skills grouped by category, used to perform
lexicon-based Named-Entity-style extraction of skills from free text
(resumes and job descriptions). Multi-word phrases are matched first so
that terms like "machine learning" are not lost to tokenization.

Each skill maps to a canonical display name plus a list of surface-form
aliases/regex-safe variants that might appear in real text.
"""

SKILL_TAXONOMY = {
    "Programming Languages": {
        "Python": ["python"],
        "Java": [r"\bjava\b(?!\s*script)"],
        "JavaScript": ["javascript", "js(?!on)"],
        "TypeScript": ["typescript"],
        "C#": ["c#", "c sharp"],
        "C++": ["c\\+\\+"],
        "SQL": [r"\bsql\b"],
        "PHP": [r"\bphp\b"],
        "Ruby": [r"\bruby\b"],
        "R": [r"\br programming\b", r"\br studio\b"],
        "Go": [r"\bgolang\b"],
        "Scala": ["scala"],
        "Bash/Shell": ["bash script", "shell script"],
        "VBA": [r"\bvba\b"],
    },
    "Web & Frameworks": {
        ".NET": [r"\.net", "dot net"],
        "ASP.NET": ["asp.net"],
        "React": [r"\breact(\.js)?\b"],
        "Angular": ["angular"],
        "Node.js": ["node.js", "nodejs"],
        "Django": ["django"],
        "Flask": ["flask"],
        "Spring": ["spring boot", "spring framework"],
        "HTML/CSS": ["html", "css"],
        "REST APIs": ["rest api", "restful"],
        "Microservices": ["microservices"],
    },
    "Data & Databases": {
        "SQL Server": ["sql server", "mssql"],
        "MySQL": ["mysql"],
        "PostgreSQL": ["postgresql", "postgres"],
        "Oracle DB": ["oracle database", "oracle db"],
        "MongoDB": ["mongodb"],
        "NoSQL": ["nosql"],
        "Data Warehousing": ["data warehous"],
        "ETL": [r"\betl\b"],
        "Power BI": ["power bi"],
        "Tableau": ["tableau"],
        "Excel": ["excel", "pivot table"],
        "Google Sheets": ["google sheets"],
        "Power Query": ["power query"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "Scikit-learn": ["scikit-learn", "sklearn"],
        "TensorFlow": ["tensorflow"],
        "Data Visualization": ["data visualization"],
        "Statistical Analysis": ["statistical analysis", "statistics"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure"],
        "Google Cloud": ["google cloud", "gcp"],
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "k8s"],
        "CI/CD": ["ci/cd", "continuous integration"],
        "Jenkins": ["jenkins"],
        "Terraform": ["terraform"],
        "Git": [r"\bgit\b", "github", "version control"],
        "Linux": ["linux", "unix"],
        "Virtualization": ["vmware", "virtualization", "hyper-v"],
    },
    "Networking & Systems": {
        "Active Directory": ["active directory"],
        "Windows Server": ["windows server"],
        "Networking (LAN/WAN)": ["lan/wan", "\\blan\\b", "\\bwan\\b", "tcp/ip"],
        "DNS/DHCP": ["dns", "dhcp"],
        "Firewalls": ["firewall"],
        "VPN": ["vpn"],
        "Remote Desktop Tools": ["remote desktop", "remote support"],
        "LANDesk": ["landesk"],
        "Citrix": ["citrix"],
        "System Administration": ["system administration", "sysadmin"],
        "Data Center Operations": ["data center"],
    },
    "Security": {
        "Cyber Security": ["cyber security", "cybersecurity", "information security"],
        "Penetration Testing": ["penetration testing", "pen testing"],
        "Vulnerability Assessment": ["vulnerability assessment", "vulnerability scan"],
        "SIEM": [r"\bsiem\b"],
        "Risk Assessment": ["risk assessment"],
        "Compliance (SOX/HIPAA/PCI)": ["hipaa", "pci-dss", "\\bsox\\b", "compliance"],
        "Incident Response": ["incident response"],
    },
    "IT Support & Help Desk": {
        "Call Tracking Software": ["call tracking"],
        "Ticketing Systems": ["ticketing system", "service now", "servicenow", "jira service"],
        "Hardware Troubleshooting": ["hardware troubleshoot", "pc repair"],
        "Software Troubleshooting": ["software troubleshoot"],
        "Customer Support": ["customer support", "customer service"],
        "Microsoft Office Suite": ["microsoft office", "ms office"],
        "Phone-based Support": ["phone support", "phone-based", "phone based technical support"],
    },
    "Project & Process": {
        "Agile": ["agile"],
        "Scrum": ["scrum"],
        "Kanban": ["kanban"],
        "Project Management": ["project management"],
        "Stakeholder Management": ["stakeholder management"],
        "Requirements Gathering": ["requirements gathering", "business requirements"],
        "Budgeting": ["budget management", "budgeting"],
        "PMP": [r"\bpmp\b"],
        "Six Sigma": ["six sigma"],
    },
    "QA & Testing": {
        "Manual Testing": ["manual testing"],
        "Automated Testing": ["automated testing", "test automation"],
        "Selenium": ["selenium"],
        "Test Case Design": ["test case", "test plan"],
        "Regression Testing": ["regression testing"],
        "Defect Tracking": ["defect tracking", "bug tracking"],
        "Load/Performance Testing": ["load testing", "performance testing"],
    },
    "NLP / Machine Learning": {
        "Machine Learning": ["machine learning"],
        "Natural Language Processing": ["natural language processing", r"\bnlp\b"],
        "spaCy": ["spacy"],
        "NLTK": ["nltk"],
        "Deep Learning": ["deep learning"],
        "Forecasting": ["forecasting", "time series"],
        "Classification Models": ["classification model"],
    },
    "Soft Skills": {
        "Communication": ["communication skills"],
        "Problem Solving": ["problem solving", "problem-solving"],
        "Teamwork": ["teamwork", "team player"],
        "Leadership": ["leadership"],
        "Time Management": ["time management"],
        "Documentation": ["documentation", "problem documentation"],
    },
}


def _flatten():
    flat = {}
    for category, skills in SKILL_TAXONOMY.items():
        for canonical, patterns in skills.items():
            flat[canonical] = {"category": category, "patterns": patterns}
    return flat


FLAT_SKILLS = _flatten()
ALL_SKILL_NAMES = sorted(FLAT_SKILLS.keys())
