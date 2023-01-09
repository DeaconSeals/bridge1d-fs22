import pytest, random, os, sys, inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from linear_genotype import LinearGenotype

def random_pop(mu, *args, **kwargs):
    return LinearGenotype.initialization(mu, *args, **kwargs)
    
def random_fitness(pop, minimum=-1000, maximum=1000):
    for i in range(len(pop)):
        pop[i].fitness = random.uniform(minimum, maximum)
    
def standard_fitness(pop, minimum=-1000, maximum=1000):
    fitness = random.uniform(minimum, maximum)
    for i in range(len(pop)):
        pop[i].fitness = fitness

def all_ones(size):
    ind = LinearGenotype()
    ind.gene = [(1, 1) for _ in range(size)]
    return ind

def all_zeroes(size):
    ind = LinearGenotype()
    ind.gene = [(0, 0) for _ in range(size)]
    return ind

def get_cross_points(gene):
    try:
        crosses = []
        run = gene[0]
        for i in range(1, len(gene)):
            if gene[i] != run:
                run = gene[i]
                crosses.append(i)
    except: #numpy nonsense
        crosses = []
        run = gene[0]
        for i in range(1, len(gene)):
            if not np.array_equal(gene[i], run):
                run = gene[i]
                crosses.append(i)
    return crosses

def same_object(obj1, obj2):
    if dir(obj1) != dir(obj2):
        return False
    dir1 = obj1.__dict__
    dir2 = obj2.__dict__
    for attr in dir1:
        try:
            if dir1[attr] != dir2[attr]:
                return False
        except: #numpy nonsense
            if not np.array_equal(dir1[attr], dir2[attr]):
                return False
    return True

def distance(genes1, genes2):
    assert len(genes1) == len(genes2)
    try:
        diff = 0
        for i in range(len(genes1)):
            if genes1[i] != genes2[i]:
                diff += 1
    except: #numpy nonsense
        diff = 0
        for i in range(len(genes1)):
            if not np.array_equal(genes1[i], genes2[i]):
                diff += 1
    return diff