from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from nlp import (
    extract_resume_text,
    extract_skills,
    calculate_match,
    get_top_matches,
    internships
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    filename = f"uploaded_{file.filename}"

    with open(filename, "wb") as f:
        f.write(await file.read())

    resume_text = extract_resume_text(filename)
    skills = extract_skills(resume_text)

    return {
        "resume_text": resume_text,
        "extracted_skills": skills
    }

@app.post("/match")
async def match(data: dict):
    resume_text = data.get("resume_text")

    if not resume_text:
        return {"error": "resume_text is required"}

    top_matches = get_top_matches(resume_text)
    return {"results": top_matches}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
