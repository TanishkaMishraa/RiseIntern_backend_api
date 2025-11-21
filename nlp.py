import pdfplumber
import docx
from sentence_transformers import SentenceTransformer, util

# Load the SBERT model only once
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Resume text extraction
# -----------------------------
def extract_resume_text(file_path):
    """Extract text from PDF or DOCX."""
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text

    elif file_path.endswith(".docx"):
        doc_obj = docx.Document(file_path)
        return "\n".join([p.text for p in doc_obj.paragraphs])

    return "Unsupported file format"

# -----------------------------
# Skill extraction
# -----------------------------
SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "excel", "pandas", "numpy", 
    "machine learning", "deep learning", "nlp", "html", "css", 
    "javascript", "react", "node", "flask", "django", 
    "power bi", "data analysis", "data science"
]

def extract_skills(text):
    """Extract known skills from resume text."""
    text = text.lower()
    found = []

    for skill in SKILL_KEYWORDS:
        if skill in text:
            found.append(skill)

    return list(set(found))

# -----------------------------
# Internship Dataset
# -----------------------------
internships = [
    {
        "title": "Machine Learning Intern",
        "description": "Looking for an intern skilled in Python, ML algorithms, data cleaning, and deep learning.",
        "skills_required": ["python", "machine learning", "deep learning", "pandas"]
    },
    {
        "title": "Frontend Developer Intern",
        "description": "Intern must know React, JavaScript, HTML, CSS, and frontend development.",
        "skills_required": ["react", "javascript", "html", "css"]
    },
    {
        "title": "Data Analyst Intern",
        "description": "Experience with SQL, Excel, Python, dashboards and data visualization preferred.",
        "skills_required": ["sql", "excel", "python", "data analysis"]
    },
    {
        "title": "NLP Research Intern",
        "description": "Knowledge of NLP, transformers, deep learning, and Python is required.",
        "skills_required": ["nlp", "python", "deep learning"]
    },
    {
        "title": "Backend Developer Intern",
        "description": "Looking for backend developer with Python, Flask, APIs, and database management skills.",
        "skills_required": ["python", "flask", "api", "database"]
    }
]

# -----------------------------
# Matching Functions
# -----------------------------
def calculate_match(resume_text, internship):
    """Compute semantic similarity + skill match + final score."""
    resume_vec = model.encode(resume_text)
    intern_vec = model.encode(internship["description"])

    semantic_score = float(util.cos_sim(resume_vec, intern_vec))

    student_skills = extract_skills(resume_text)
    required = internship["skills_required"]

    skill_match = (
        len(set(student_skills) & set(required)) / len(required)
        if required else 0
    )

    final_score = 0.65 * semantic_score + 0.35 * skill_match

    return {
        "title": internship["title"],
        "description": internship["description"],
        "required_skills": required,
        "extracted_skills": student_skills,
        "semantic_score": semantic_score,
        "skill_match": skill_match,
        "final_score": final_score,
    }

def get_top_matches(resume_text, limit=5):
    results = [calculate_match(resume_text, i) for i in internships]
    return sorted(results, key=lambda x: x["final_score"], reverse=True)[:limit]
