from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired

def get_representation_models():
    return [MaximalMarginalRelevance(diversity=0.3), KeyBERTInspired()]
