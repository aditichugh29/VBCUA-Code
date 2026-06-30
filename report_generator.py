from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(data, buffer):
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add Title and Data
    story.append(Paragraph("VBCUA Evaluation Report", styles['Title']))
    story.append(Paragraph(f"Final Score: {data['score']:.2f}", styles['Normal']))
    story.append(Paragraph(f"Understanding Level: {data['level']}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add Full Transcription
    story.append(Paragraph("AI Transcription:", styles['Heading2']))
    story.append(Paragraph(data.get('transcription', 'No transcription available'), styles['Normal']))
    
    doc.build(story)
