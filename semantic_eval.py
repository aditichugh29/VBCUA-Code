from sentence_transformers import SentenceTransformer, util

def calculate_similarity(student_text, reference_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(student_text)
    embedding2 = model.encode(reference_text)
    score = util.cos_sim(embedding1, embedding2)
    return float(score.item())