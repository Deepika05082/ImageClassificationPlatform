from src.inference.utils import label_map

def test_label_map():
    assert label_map(0.7) == "dog"
    assert label_map(0.3) == "cat"
