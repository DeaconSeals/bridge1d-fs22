from test_utils import *
import random, pytest, copy, os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from snake_eyes import read_config
from selection import *
import base_evolution as evo

config = read_config('./configs/green1b_config.txt',\
                    globalVars=globals(), localVars=locals())
iterations = 25

class TestChildGeneration:
    def test_length(self):
        ea = evo.BaseEvolutionPopulation(**config['EA_configs'], **config)
        for _ in range(iterations):
            ea.mu = random.randint(5, 1000)
            ea.parent_selection_kwargs['k'] = ea.mu / 2
            config['parent_selection_kwargs']['k'] = min(config['parent_selection_kwargs']['k'], ea.mu)
            ea.population = random_pop(ea.mu, **config['initialization_kwargs'])
            random_fitness(ea.population)
            ea.num_children = random.randint(1, 500)
            assert len(ea.generate_children()) == ea.num_children
    
    def test_unmodified_parents(self):
        #child generation has no impact on the population
        ea = evo.BaseEvolutionPopulation(**config['EA_configs'], **config)
        for _ in range(iterations):
            ea.mu = random.randint(5, 1000)
            ea.parent_selection_kwargs['k'] = ea.mu / 2
            ea.population = random_pop(ea.mu, **config['initialization_kwargs'])
            random_fitness(ea.population)
            ea.num_children = random.randint(1, 500)
            copies = copy.deepcopy(ea.population)
            children = ea.generate_children()
            for i in range(len(copies)):
                assert same_object(ea.population[i], copies[i])