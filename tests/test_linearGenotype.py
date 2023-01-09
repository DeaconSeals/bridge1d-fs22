from test_utils import *
import random, pytest, copy
import numpy as np
from snake_eyes import read_config

config = read_config('./configs/green1b_config.txt',\
                    globalVars=globals(), localVars=locals())
iterations = 250

class TestUniformRecombination:
    @classmethod
    def setup_class(cls):
        config['recombination_kwargs']['method'] = 'uniform'
    
    def test_length(self):
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            parents = random_pop(2, **config['initialization_kwargs'])
            child = parents[0].recombine(parents[1], **config['recombination_kwargs'])
            assert len(child.gene) == config['initialization_kwargs']['length']

    def test_is_uniform(self):
        #each locus is selected uniform randomly
        expected_lower_bound = 0.2
        expected_upper_bound = 0.8
        out_of_bounds = 0
        m_iterations = iterations * 100
        boardsize = random.randrange(10, 100)
        ones = all_ones(boardsize)
        zeroes = all_zeroes(boardsize)
        hits = [0 for _ in range(boardsize)]
        for _ in range(m_iterations):
            if random.randint(0, 1) == 0:
                child = ones.recombine(zeroes, **config['recombination_kwargs'])
            else:
                child = zeroes.recombine(ones, **config['recombination_kwargs'])
            num_ones = 0
            for i in range(len(child.gene)):
                was_all_ones = True
                for sub in child.gene[i]:
                    if sub != 1:
                        was_all_ones = False
                if was_all_ones:
                    num_ones += 1
                    hits[i] += 1
            ratio = num_ones / boardsize
            if ratio < expected_lower_bound or ratio > expected_upper_bound:
                out_of_bounds += 1
        assert out_of_bounds < m_iterations / 10
        min_bound = 0.4
        max_bound = 0.6
        for hit in hits:
            ratio = hit / m_iterations
            assert ratio < max_bound
            assert ratio > min_bound

    def test_parents_unmodified(self):
        #parents are not modified at all by recombination
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            parents = random_pop(2, **config['initialization_kwargs'])
            random_fitness(parents)
            copies = [copy.deepcopy(x) for x in parents]
            child = parents[0].recombine(parents[1], **config['recombination_kwargs'])
            assert same_object(copies[0], parents[0])
            assert same_object(copies[1], parents[1])

class Test1PointCrossoverRecombination:
    @classmethod
    def setup_class(cls):
        config['recombination_kwargs']['method'] = '1-point crossover'
        
    def test_length(self):
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            parents = random_pop(2, **config['initialization_kwargs'])
            child = parents[0].recombine(parents[1], **config['recombination_kwargs'])
            assert len(child.gene) == config['initialization_kwargs']['length']

    def test_has_1_point(self):
        #the resulting child has one distinct crossover point
        for _ in range(iterations):
            boardsize = random.randrange(10, 100)
            ones = all_ones(boardsize)
            zeroes = all_zeroes(boardsize)
            if random.randint(0, 1) == 0:
                child = ones.recombine(zeroes, **config['recombination_kwargs'])
            else:
                child = zeroes.recombine(ones, **config['recombination_kwargs'])
            assert len(get_cross_points(child.gene)) == 1

    def test_point_is_uniformly_selected(self):
        #the crossover point is selected uniformly out of all valid loci
        boardsize = random.randrange(10, 100)
        hits = {x:0 for x in range(boardsize)}
        m_iterations = iterations * 100
        ones = all_ones(boardsize)
        zeroes = all_zeroes(boardsize)
        for _ in range(m_iterations):
            if random.randint(0, 1) == 0:
                child = ones.recombine(zeroes, **config['recombination_kwargs'])
            else:
                child = zeroes.recombine(ones, **config['recombination_kwargs'])
            hits[get_cross_points(child.gene)[0]] += 1
        expected = m_iterations / (boardsize - 1)
        expected_lower_bound = expected - (expected * 0.2)
        expected_upper_bound = expected + (expected * 0.2)
        min_bound = expected - (expected * 0.4)
        max_bound = expected + (expected * 0.4)
        assert hits[0] == 0
        hits[0] = expected
        out_of_bounds = 0
        for hit in hits.values():
            assert hit < max_bound
            assert hit > min_bound
            if hit > expected_upper_bound or hit < expected_lower_bound:
                out_of_bounds += 1
        assert out_of_bounds < boardsize / 20

    def test_parents_unmodified(self):
        #parents are not modified at all by recombination
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            parents = random_pop(2, **config['initialization_kwargs'])
            random_fitness(parents)
            copies = [copy.deepcopy(x) for x in parents]
            child = parents[0].recombine(parents[1], **config['recombination_kwargs'])
            assert same_object(copies[0], parents[0])
            assert same_object(copies[1], parents[1])

class TestMutation:
    def test_length(self):
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            parents = random_pop(1, **config['initialization_kwargs'])
            random_fitness(parents)
            child = parents[0].mutate(**config['mutation_kwargs'])
            assert len(child.gene) == config['initialization_kwargs']['length']

    def test_sometimes_changes(self):
        #mutation produces changes
        passed = False
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            parents = random_pop(1, **config['initialization_kwargs'])
            random_fitness(parents)
            child = parents[0].mutate(**config['mutation_kwargs'])
            if distance(child.gene, parents[0].gene) > 0:
                passed = True
                break
        assert passed

    def test_no_locus_bias(self):
        #each locus has the same independent chance of mutating
        config['initialization_kwargs']['length'] = random.randrange(10, 100)
        hits = [0 for _ in range(config['initialization_kwargs']['length'])]
        m_iterations = iterations * 100
        for _ in range(m_iterations):
            parents = random_pop(1, **config['initialization_kwargs'])
            random_fitness(parents)
            child = parents[0].mutate(**config['mutation_kwargs'])
            try:
                for i in range(config['initialization_kwargs']['length']):
                    if parents[0].gene[i] != child.gene[i]:
                        hits[i] += 1
            except: #numpy nonsense
                for i in range(config['initialization_kwargs']['length']):
                    if not np.array_equal(parents[0].gene[i], child.gene[i]):
                        hits[i] += 1
        average_change = sum(hits) / len(hits)
        expected_lower_bound = average_change - average_change * 0.2
        expected_upper_bound = average_change + average_change * 0.2
        out_of_bounds = 0
        for locus in hits:
            if locus > expected_upper_bound or locus < expected_lower_bound:
                out_of_bounds += 1
        assert out_of_bounds < config['initialization_kwargs']['length'] / 20