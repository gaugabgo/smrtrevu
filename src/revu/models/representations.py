from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    
def get_representation_models():
    """
    Get representation models in dictionary format for clarity.
    Models are applied in the order specified.
    """
    return {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.5)
    }