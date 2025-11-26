<h1 align="center">🔎 ResearchReach: Intelligent Research Paper Matcher & Cold-Email Assistant</h1>
<p align="center"><i>AI-powered system that analyzes resumes and recommends the most relevant research papers</i></p>

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

## 📑 Table of Contents

- [🔎 ResearchReach: Intelligent Research Paper Matcher & Cold-Email Assistant](#-researchreach-intelligent-research-paper-matcher--cold-email-assistant)
- [🌟 Overview](#-overview)
- [🚀 Introduction](#-introduction)
- [🔍 How It Works](#-how-it-works)
  - [1️⃣ Resume Data Extraction](#1️⃣-resume-data-extraction)
- [🔍 Research Paper Matching System](#-research-paper-matching-system)
  - [🛠️ Tech Stack](#-tech-stack)
  - [🏆 Features](#-features)
  - [🚀 Process Overview](#-process-overview)
- [📝 1️⃣ Resume Parsing and Skill Extraction](#-1️⃣-resume-parsing-and-skill-extraction)
- [📜 2️⃣ Research Paper Retrieval](#-2️⃣-research-paper-retrieval)
- [🔎 3️⃣ Convert to Sentence Embeddings](#-3️⃣-convert-to-sentence-embeddings)
- [📈 4️⃣ Compute Cosine Similarity](#-4️⃣-compute-cosine-similarity)
- [💡 5️⃣ Final Output](#-5️⃣-final-output)
- [✉️ 6️⃣ Email Generation](#️-6️⃣-email-generation)
- [🤝 Contributors](#-contributors)


## 🌟 Overview

**ResearchReach** is an AI-driven web platform that intelligently matches research papers with a candidate’s resume.  
Using advanced natural language processing techniques such as **Sentence-BERT (SBERT) embeddings** and **cosine similarity**, the system evaluates a user’s:

- Skills  
- Projects  
- Technical experience  
- Research interests  

It then identifies the most relevant research papers from the web and automatically drafts a professional cold-email tailored to the selected paper.

This makes the process of research discovery and outreach faster, more accurate, and significantly more efficient.

---

## 🚀 Introduction

Finding research papers that align precisely with your skills and academic profile can be tedious.  
**ResearchReach** fully automates this process in four steps:

- ✅ Extracts skills and project details from the resume  
- ✅ Converts resume and paper text into embeddings using SBERT  
- ✅ Computes semantic similarity using cosine similarity  
- ✅ Recommends the most relevant research paper with a high matching score  

Designed for students, researchers, and applicants seeking internships or collaboration opportunities, ResearchReach offers a streamlined, intelligent, and user-friendly experience.  

---

## 🔍 **How It Works**  
The matching process follows a **4-step pipeline**:  

### 1️⃣ **Resume Data Extraction**  
- The system extracts key details from the candidate's resume, including:  
   - ✅ **Skills** (e.g., Machine Learning, NLP)  
   - ✅ **Projects** (e.g., Fake News Detection using BERT)  

✅ Example:  
```python
Skills = ["Machine Learning", "Natural Language Processing", "Deep Learning", "Python"]  
Projects = ["Fake News Detection using BERT", "Text Summarization with LSTM"]  
```
This information is concatenated into a single text input:
```python
"Machine Learning Natural Language Processing Deep Learning Python Fake News Detection using BERT Text Summarization with LSTM" 
```

# 🔍 **Research Paper Matching System**

## 🛠️ **Tech Stack**
| Component              | Tool                                      |
|----------------------- |------------------------------------------|
| **Frontend**            | React.js                                  |
| **Backend**             | Flask                                     |
| **Embedding Model**     | Sentence-BERT (all-MiniLM-L6-v2)         |
| **Paper Retrieval**     | Semantic Scholar API                      |
| **Similarity Calculation** | Cosine Similarity (Scikit-learn)         |
| **Email Generation**     | Gemini API                                |
| **Paper Download**       | Unpaywall API                             |

---

## 🏆 **Features**
✅ Fast and Efficient: Handles large datasets quickly using SBERT.  
✅ Accurate Matching: High similarity scoring using cosine similarity.  
✅ Automated Paper Retrieval: Uses Semantic Scholar to find relevant papers.  
✅ Secure Data Handling: Ensures data privacy and integrity.  
✅ Email Automation: Automatically generates internship request emails based on the matching paper.  

---

## 🚀 **Process Overview**
1. **Resume Parsing and Skill Extraction**  
2. **Research Paper Retrieval**  
3. **Convert to Sentence Embeddings**  
4. **Compute Cosine Similarity**  
5. **Generate and Send Email**  

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

## 📝 **1️⃣ Resume Parsing and Skill Extraction**
The system extracts skills and projects from the resume using `pdfplumber`, `spaCy`, and `KeyBERT`.

**Example Skills:**  
`Machine Learning, Natural Language Processing, Deep Learning, Python, Fake News Detection using BERT, Text Summarization with LSTM`

---

## 📜 **2️⃣ Research Paper Retrieval**
The system retrieves research papers using Web Scraping with the help of beautifulsoup4 & Spacy 

**Example papers:**  

**📜 Paper 1:**  
**Title:** "A Deep Learning Approach to Fake News Detection"  
**Abstract:** "We propose a model based on BERT for detecting fake news articles. Our approach achieves state-of-the-art performance in text classification tasks."  

**📜 Paper 2:**  
**Title:** "Efficient Image Classification with CNNs"  
**Abstract:** "We present an optimized CNN model for image classification. The model reduces computational cost while maintaining accuracy."  

---

## 🔎 **3️⃣ Convert to Sentence Embeddings**
The system converts text into high-dimensional vector embeddings using **Sentence-BERT** (`all-MiniLM-L6-v2`):

```python
from sentence_transformers import SentenceTransformer  

embed_model = SentenceTransformer('all-MiniLM-L6-v2')  
resume_embedding = embed_model.encode(resume_text)  
paper_1_embedding = embed_model.encode(paper_1_text)  
paper_2_embedding = embed_model.encode(paper_2_text)  
```
## ✅ Example vector embeddings:
```css
Resume Embedding → [0.12, -0.08, ..., 0.32]  
Paper 1 Embedding → [0.11, -0.07, ..., 0.30]  
Paper 2 Embedding → [0.02, 0.45, ..., -0.12]  
```
---
## 📈 4️⃣ Compute Cosine Similarity  
Cosine similarity measures how similar two vectors are:

\[
\text{Cosine Similarity} = \frac{A \cdot B}{||A|| \cdot ||B||}
\]


---

✅ **Example calculation:**  
```python
from sklearn.metrics.pairwise import cosine_similarity  

similarity_1 = cosine_similarity([resume_embedding], [paper_1_embedding])  
similarity_2 = cosine_similarity([resume_embedding], [paper_2_embedding])  
```

| **Pair**               | **Similarity Score** | **Result**              |
|-----------------------|----------------------|-------------------------|
| **Resume & Paper 1**  | **0.92**              | ✅ High Similarity       |
| **Resume & Paper 2**  | **0.34**              | ❌ Low Similarity        |

---

## 💡 5️⃣ Final Output  
The paper with the highest similarity score is selected as the most relevant match.

✅ **Most Relevant Paper Found!**  
**Title:** *"A Deep Learning Approach to Fake News Detection"*  
**Abstract:** *"We propose a model based on BERT for detecting fake news articles. Our approach achieves state-of-the-art performance in text classification tasks."*  
**Similarity Score:** **0.92**  

---

## ✉️ 6️⃣ Email Generation  
Once a matching paper is found, the system generates an internship request email using the **Gemini API**.

### **Template Options:**  
✅ Formal & Professional  
✅ Technical & Research-Oriented  
✅ Enthusiastic & Passionate  

---
## 🤝 **Contributors**  
We would like to extend our heartfelt gratitude to everyone who contributed to this project. Your hard work and dedication made this possible!  

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/125748305?v=4" width="80" height="80" alt="Your Name">
      <br>
      <a href="https://github.com/Srujanrana07"><b>Srujan Rana</b></a>
      <br>
      🏆 Project Lead, Backend Developer
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/119315259?v=4" width="80" height="80" alt="Contributor 1">
      <br>
      <a href="https://github.com/contributor1"><b>Rudra Prasad Jena</b></a>
      <br>
      💻 Frontend Developer & 🌐 API Integration
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/161008301?v=4" width="80" height="80" alt="Contributor 1">
      <br>
      <a href="https://github.com/Abhishek-ro"><b>Abhishek Kumar</b></a>
      <br>
      💻 Frontend Developer
    </td>
  </tr>
</table>

🌟 **Want to contribute?**  
We welcome contributions from the community! If you'd like to improve the project or report issues, feel free to fork the repo and submit a pull request.  

