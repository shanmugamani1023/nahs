from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

for section in doc.sections:
    section.top_margin = Inches(0.3)
    section.bottom_margin = Inches(0.3)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)

from docx.shared import Pt
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(9)

def add_name(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(20)

def add_contact(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(9)

def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0, 0, 0)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '000000')
    pBdr.append(bottom)
    pPr.append(pBdr)

def add_paragraph(doc, text, bold=False, size=9.5, space_before=0, space_after=2):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p

def add_bullet(doc, text, bold_part=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.left_indent = Inches(0.15)
    if bold_part:
        run1 = p.add_run(bold_part)
        run1.bold = True
        run1.font.size = Pt(9.5)
        run2 = p.add_run(text)
        run2.font.size = Pt(9.5)
    else:
        run = p.add_run(text)
        run.font.size = Pt(9.5)

def add_job_header(doc, company, role, date):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(1)
    run1 = p.add_run(f"{company} — {role}")
    run1.bold = True
    run1.font.size = Pt(9.5)
    run2 = p.add_run(f"    {date}")
    run2.bold = False
    run2.font.size = Pt(9.5)

# NAME & CONTACT
add_name(doc, "SHANMUGAMANI")
add_contact(doc, "+91-7402320768  |  shanmugamanimeyialagan@gmail.com  |  github.com/shanmugamani1023  |  linkedin.com/in/shanmugamani/")

# SUMMARY
add_section_heading(doc, "SUMMARY")
add_paragraph(doc, "AI and Computer Vision Engineer with 4+ years of experience designing and deploying production-ready Computer Vision and AI systems. Expert in image recognition, object detection, video analysis, and image processing using deep learning and neural network architectures. Proven ability to build scalable AI pipelines for cloud and on-premises environments, implement Generative AI models, and deliver high-accuracy solutions in autonomous retail applications.", space_after=2)

# TECHNICAL SKILLS
add_section_heading(doc, "TECHNICAL SKILLS")
skills = [
    ("Programming Languages: ", "Python, SQL"),
    ("Computer Vision: ", "Object Detection, Image Recognition, Image Segmentation, Video Analysis, Pose Estimation, OCR, Image Processing, YOLO, OpenCV, SAM"),
    ("Deep Learning and Neural Networks: ", "CNN, Transfer Learning, PyTorch, TensorFlow, Keras, ResNet, YOLOX"),
    ("Generative AI and LLMs: ", "LangChain, LangGraph, RAG Pipeline, AI Agents, Prompt Engineering, Fine-tuning (LoRA, QLoRA), Multimodal LLMs"),
    ("AI Pipeline and Deployment: ", "FastAPI, REST APIs, Docker, Azure, On-premises Edge Deployment, Model Optimization"),
    ("Vector Databases: ", "Qdrant, ChromaDB, FAISS"),
    ("Databases and Tools: ", "PostgreSQL, Redis, Git, LangSmith, Pandas, NumPy, Scikit-learn"),
]
for bold_part, normal_part in skills:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    r1 = p.add_run(bold_part)
    r1.bold = True
    r1.font.size = Pt(9.5)
    r2 = p.add_run(normal_part)
    r2.font.size = Pt(9.5)

# EXPERIENCE
add_section_heading(doc, "EXPERIENCE")

# Digit7
add_job_header(doc, "Digit7", "Senior Software Engineer", "12/2024 - Present")

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(1)
r = p.add_run("Product Handler — Real-time Product Detection System:")
r.bold = True
r.font.size = Pt(9)

add_bullet(doc, "Designed and deployed scalable Computer Vision pipeline for autonomous retail: IoT trigger, RTSP video stream capture, MOG2 motion detection, SAM image segmentation, ResNet image classification, Redis storage for automated billing; improved model accuracy from 80% to 95%.")
add_bullet(doc, "Built production-ready multi-threaded video analysis system supporting 10+ concurrent RTSP camera streams; evaluated model performance using precision, recall, and F1 metrics; maintained 95%+ accuracy across 4 shelf types.")
add_bullet(doc, "OpenCV, SAM, ResNet, PyTorch, PyAV, Redis, Docker, Azure", bold_part="Tools: ")

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(1)
r = p.add_run("Pose Estimator — Customer Identification System:")
r.bold = True
r.font.size = Pt(9)

add_bullet(doc, "Built real-time deep learning pipeline for customer identification: YOLOX person detection, Shapely polygon matching, homography calibration to map floor coordinates to camera pixel space for automated billing in autonomous retail.")
add_bullet(doc, "OpenCV, YOLOX, Shapely, PyAV, Redis, PyTorch", bold_part="Tools: ")

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(1)
r = p.add_run("Retail AI Chatbot — Generative AI Application:")
r.bold = True
r.font.size = Pt(9)

add_bullet(doc, "Built and deployed Generative AI application using LangChain agent, RAG pipeline (Qdrant vector database + PostgreSQL), scalable REST API (FastAPI) with JWT auth, Redis memory, and three-tier LLM fallback for production reliability.")
add_bullet(doc, "LangChain, LangGraph, Qdrant, FastAPI, Redis, PostgreSQL, Docker, Azure, LangSmith", bold_part="Tools: ")

# Shrav Infotech
add_job_header(doc, "Shrav Infotech", "Software Engineer", "12/2021 - 11/2024")
add_bullet(doc, "Reduced air filter inspection time from 3 min to 3 sec (95% accuracy) using OpenCV image processing pipeline.", bold_part="Air-filter Inspection: ")
add_bullet(doc, "Improved image proofing speed from 2 min to 3 sec per image (96% OCR, 94% object detection) using OCR + YOLO pipeline.", bold_part="Image Proofing System: ")
add_bullet(doc, "Deployed OCR system to detect engraved part numbers across LH/RH components at production scale.", bold_part="Engraved Text Detection: ")

# EDUCATION
add_section_heading(doc, "EDUCATION")
add_bullet(doc, "B.E Mechanical Engineering, CGPA 7.6, Dhirajlal Gandhi College of Technology, 2017, Salem.")

# Save
doc.save("/home/digit7/shanmugamani/nahs/acc_resume.docx")
print("Done")
