# ResearchReach

AI-powered research paper recommendation and cold-email generation platform using semantic similarity and transformer embeddings.

---
<p align="center">
  <!-- Technology Badges -->
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/Flask-Backend-000000?style=flat&logo=flask" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=flat&logo=react" />
  <img src="https://img.shields.io/badge/SBERT-Embeddings-green?style=flat" />
  <img src="https://img.shields.io/badge/Cosine%20Similarity-ML%20Model-orange?style=flat" />
  <img src="https://img.shields.io/badge/Gemini-Email%20Generation-4285F4?style=flat&logo=google" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

---


## Overview

ResearchReach is an intelligent research assistance platform that analyzes a candidate’s resume and recommends highly relevant research papers based on semantic similarity. The system leverages transformer-based embeddings and natural language processing techniques to match user profiles with research publications and automatically generate professional cold emails for research outreach.

The platform is designed for:
- Students seeking research internships
- Researchers exploring relevant publications
- Professionals looking for academic collaborations
- Applicants preparing outreach emails for professors and labs

---

## Key Features

- Resume parsing and skill extraction
- Semantic research paper recommendation
- SBERT-based embedding generation
- Cosine similarity ranking system
- Automated research paper retrieval
- AI-generated professional cold emails
- Web-based user interface using React and Flask

---

## System Architecture

The platform follows a multi-stage NLP pipeline:
<img width="400" height="600" alt="Gemini_Generated_Image_2mojsp2mojsp2moj" src="https://github.com/user-attachments/assets/b6d3ca8f-55ad-4699-98fd-d47f62b99b94" />
---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React.js |
| Backend | Flask |
| NLP Framework | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Similarity Metric | Cosine Similarity |
| Resume Parsing | pdfplumber, spaCy, KeyBERT |
| Research Paper Retrieval | Semantic Scholar API |
| Email Generation | Gemini API |
| Machine Learning Libraries | Scikit-learn |

---

## Workflow

### 1. Resume Parsing

The system extracts technical information from uploaded resumes, including:
- Skills
- Projects
- Research interests
- Technical domains

Example extracted content:

```python
skills = [
    "Machine Learning",
    "Natural Language Processing",
    "Deep Learning",
    "Python"
]

projects = [
    "Fake News Detection using BERT",
    "Text Summarization with LSTM"
]
```

---

### 2. Research Paper Retrieval

Relevant research papers are collected using semantic search and API-based retrieval methods.

Example paper:

**Title:**  
A Deep Learning Approach to Fake News Detection

**Abstract:**  
We propose a transformer-based architecture for detecting fake news articles using contextual embeddings and attention mechanisms.

---

### 3. Embedding Generation

The platform uses Sentence-BERT (`all-MiniLM-L6-v2`) to generate dense semantic embeddings for resumes and research paper abstracts.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

resume_embedding = model.encode(resume_text)
paper_embedding = model.encode(paper_text)
```

---

### 4. Similarity Computation

Cosine similarity is used to measure semantic relevance between the resume and research papers.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_score = cosine_similarity(
    [resume_embedding],
    [paper_embedding]
)
```

Example similarity results:

| Resume vs Paper | Similarity Score |
|---|---|
| Paper 1 | 0.92 |
| Paper 2 | 0.34 |

The paper with the highest score is recommended to the user.

---

### 5. Cold Email Generation

Once the most relevant paper is identified, the system generates a professional outreach email tailored to:
- Research domain
- User skills
- Paper topic
- Academic interest

The generated email can be used for:
- Internship applications
- Research collaborations
- Professor outreach
- Academic networking

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Srujanrana07/ResearchReach-A-Cold-Email-Helper.git

cd app #for the backend
```

---

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

### Start Backend Server

```bash
python app.py
```

---

### Start Frontend

```bash
cd frontend

npm install

npm start
```

---

## Project Structure

```bash
ResearchReach/
├── app
│   ├── .gitignore
│   ├── app.py
│   ├── email_generator.py
│   ├── email_sender.py
│   ├── paper_retrieval.py
│   ├── requirements.txt
│   ├── resume_processor.py
│   ├── similarity.py
│   └── test.py
├── front
│   ├── public
│   ├── src
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── App.test.js
│   │   ├── index.css
│   │   ├── index.js
│   │   ├── logo.svg
│   │   ├── reportWebVitals.js
│   │   ├── setupTests.js
│   │   └── temp.css
│   ├── .gitignore
│   ├── README.md
│   ├── package-lock.json
│   └── package.json
└── README.md
```

---
## Picture Based application demo

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">

  <!-- Step 1 -->
  <div style="text-align: center;">
    <img width="350" src="https://github.com/user-attachments/assets/5331eb16-33d7-4a99-8581-580cb34eb6f6" alt="Step 3"/>
    <p><b>Step 3</b></p>
  </div>
  <!-- Step 2 -->
  <div style="text-align: center;">
    <img width="350" src="https://github.com/user-attachments/assets/d6f635a8-f9da-40f6-951e-5349e92d1e57" alt="Step 2"/>
    <p><b>Step 2</b></p>
  </div>

  <!-- Step 3 -->
  <div style="text-align: center;">
    <img width="350" src="https://github.com/user-attachments/assets/f5bd782c-c16b-47a1-bda9-660b7edc8ca7" alt="Step 1"/>
    <p><b>Step 1</b></p>
  </div>

  <!-- Step 4 -->
  <div style="text-align: center;">
    <img width="350" src="https://github.com/user-attachments/assets/46fd2132-709b-464b-9210-5d8dcdb47d06" alt="Step 4"/>
    <p><b>Step 4</b></p>
  </div>

</div>

---

## Applications

ResearchReach can be used for:
- Research internship applications
- Academic paper recommendation systems
- Intelligent academic search platforms
- AI-powered networking tools
- Personalized research discovery

---

## Future Improvements

- Multi-paper recommendation engine
- Research trend analysis
- Citation-based ranking
- PDF summarization
- Research topic clustering
- User authentication and profile tracking
- Real-time recommendation updates

---

## Contributors

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/125748305?v=4" width="80" height="80" alt="Your Name">
      <br>
      <a href="https://github.com/Srujanrana07"><b>Srujan Rana</b></a>
      <br>
      Project Lead, Backend Developer
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/119315259?v=4" width="80" height="80" alt="Contributor 1">
      <br>
      <a href="https://github.com/contributor1"><b>Rudra Prasad Jena</b></a>
      <br>
      Frontend Developer & API Integration
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/161008301?v=4" width="80" height="80" alt="Contributor 1">
      <br>
      <a href="https://github.com/Abhishek-ro"><b>Abhishek Kumar</b></a>
      <br>
      Frontend Developer
    </td>
  </tr>
</table>

---

## Contributing

Contributions are welcome.  
Feel free to fork the repository, open issues, or submit pull requests to improve the platform.

---

## License

This project is licensed under the MIT License.
