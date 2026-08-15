def label_map(pred):
    return "dog" if pred > 0.5 else "cat"
