from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    
def get_representation_models():
    return [MaximalMarginalRelevance(diversity=0.3), KeyBERTInspired()]
