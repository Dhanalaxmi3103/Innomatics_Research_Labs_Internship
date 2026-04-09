from fpdf import FPDF
import os

def generate_pdf(text, filename="output/article.pdf"):
    os.makedirs("output", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf.output(filename)

    return filename