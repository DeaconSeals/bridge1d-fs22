import random

# Parent selection functions---------------------------------------------------
def uniform_random_selection(population, n, **kwargs):
    # TODO: select n individuals uniform randomly
    pass

def k_tournament_with_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments with replacement to select n individuals
    pass

def fitness_proportionate_selection(population, n, **kwargs):
    # TODO: select n individuals using fitness proportionate selection
    pass


# Survival selection functions-------------------------------------------------
def truncation(population, n, **kwargs):
    # TODO: perform truncation selection to select n individuals
    pass

def k_tournament_without_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments without replacement to select n individuals
    #       Note: an individual should never be cloned from surviving twice!
    pass

# Yellow deliverable parent selection function---------------------------------
def stochastic_universal_sampling(population, n, **kwargs):
    # Recall that yellow deliverables are required for students in the grad
    # section but bonus for those in the undergrad section.
    # TODO: select n individuals using stochastic universal sampling
    pass
