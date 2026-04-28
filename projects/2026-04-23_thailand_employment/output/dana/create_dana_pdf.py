from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_pdf_report(input_md, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Data Cleaning Report - Dana")
    
    c.setFont("Helvetica", 12)
    y = height - 80
    with open(input_md, "r", encoding="utf-8") as f:
        for line in f:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 12)
            c.drawString(50, y, line.strip())
            y -= 20
    c.save()

# Path settings
base_path = "D:/DataScienceOS/projects/2026-04-23_thailand_employment"
report_md = os.path.join(base_path, "output/dana/dana_cleaning_report.md")
report_pdf = os.path.join(base_path, "output/dana/dana_cleaning_report.pdf")

create_pdf_report(report_md, report_pdf)
print(f"PDF created at {report_pdf}")