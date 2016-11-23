from models import get_model_by_name


def compute_score(model_name, image):
    "Accepts a model name and a PIL image and returns an integer from 0 to 4"
    model_class = get_model_by_name(model_name)
    model = model_class()
    score = model.predict(image)
    return score

