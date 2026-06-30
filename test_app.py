import pytest
from scoring_engine import classify_understanding, calculate_filler_ratio

def test_classify_understanding_returns_string():
    """Validates that the scoring engine successfully outputs a text classification."""
    # Test high, medium, and low scores
    assert isinstance(classify_understanding(0.85), str)
    assert isinstance(classify_understanding(0.45), str)
    assert isinstance(classify_understanding(0.15), str)

def test_calculate_filler_ratio_returns_float():
    """Validates that the filler word calculator successfully returns a ratio."""
    sample_text = "um basically this is like a test"
    ratio = calculate_filler_ratio(sample_text)
    
    # Ensures the output is a decimal number
    assert isinstance(ratio, float)
    
    # Ensures the ratio makes mathematical sense (between 0 and 1)
    assert 0.0 <= ratio <= 1.0