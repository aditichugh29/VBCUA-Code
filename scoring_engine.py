def classify_understanding(score):
    if score >= 0.7:
        return "Strong Understanding"
    elif score >= 0.4:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"

def calculate_filler_ratio(text):
    # Basic logic: count 'um', 'uh', 'like'
    fillers = ['um', 'uh', 'like', 'actually', 'basically']
    words = text.lower().split()
    count = sum(1 for word in words if word in fillers)
    return count / len(words) if len(words) > 0 else 0