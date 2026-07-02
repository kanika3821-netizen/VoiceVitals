from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate,
    Paragraph, Spacer, Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime

def generate_pdf_report(healthy_pct, risk_pct,
                        patient_name="Anonymous"):
    filename = (f"VoiceVitals_Report_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                f".pdf")
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "VoiceVitals Screening Report",
        styles['Title']))
    story.append(HRFlowable(width="100%"))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"<b>Patient:</b> {patient_name}",
        styles['Normal']))
    story.append(Paragraph(
        f"<b>Date:</b> "
        f"{datetime.now().strftime('%d %B %Y %H:%M')}",
        styles['Normal']))
    story.append(Spacer(1, 12))

    data = [
        ['Metric', 'Score', 'Status'],
        ['Healthy Pattern Match',
         f"{healthy_pct:.1f}%",
         'Good' if healthy_pct > 60 else 'Low'],
        ['At-Risk Pattern Match',
         f"{risk_pct:.1f}%",
         'Low' if risk_pct < 40 else 'Elevated'],
        ['Model Accuracy', '78-92%',
         'Cross-validated'],
        ['At-Risk Sensitivity', '97%',
         'Clinical dataset'],
    ]
    table = Table(data, colWidths=[
        2.5*inch, 1.5*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0),
         colors.HexColor('#1E88E5')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0),
         'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1,
         colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#EBF5FB'),
          colors.white]),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    if risk_pct > 65:
        interpretation = (
            "Your voice pattern shows significant "
            "similarity to at-risk profiles. This does "
            "NOT mean you have Parkinson's disease. "
            "Please consult a neurologist.")
    elif risk_pct > 40:
        interpretation = (
            "Your voice pattern shows some at-risk "
            "indicators. Monitor over time and consult "
            "a doctor if symptoms appear.")
    else:
        interpretation = (
            "Your voice pattern closely matches healthy "
            "profiles. Continue monitoring regularly.")

    story.append(Paragraph(
        "Interpretation", styles['Heading2']))
    story.append(Paragraph(
        interpretation, styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(HRFlowable(width="100%"))
    story.append(Spacer(1, 8))
    disc = styles['Normal'].clone('disc')
    disc.fontSize = 8
    disc.textColor = colors.grey
    story.append(Paragraph(
        "DISCLAIMER: VoiceVitals is a screening tool "
        "only — NOT a medical diagnosis. Always consult "
        "a qualified neurologist for clinical evaluation.",
        disc))

    doc.build(story)
    return filename