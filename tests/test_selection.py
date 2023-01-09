from test_utils import *
import random, pytest, copy, os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import selection as sel
from snake_eyes import read_config

config = read_config('./configs/green1b_config.txt',\
                    globalVars=globals(), localVars=locals())
iterations = 20

class TestUniformRandomParentSelection:
    def test_output_size(self):
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            outsize = random.randint(1, popsize * 2)
            assert len(sel.uniform_random_selection(pop, outsize,\
                                **config['parent_selection_kwargs'])) == outsize

    def test_is_uniform(self):
        #selection is uniformly distributed, with some noise
        popsize = random.randrange(50, 500)
        out_of_bounds = 0
        outsize = popsize * 1000
        expected = outsize / popsize
        min_bound = max(expected / 1.1, 1)
        max_bound = expected * 1.1
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            hits = {x:0 for x in pop}
            for individual in sel.uniform_random_selection(pop, outsize,\
                                **config['parent_selection_kwargs']):
                hits[individual] += 1
            for hit in hits.values():
                if min_bound > hit or max_bound < hit:
                    out_of_bounds += 1
        assert out_of_bounds <= 1.15 * iterations

    def test_definitely_duplicates(self):
        #always has duplicates when out > in
        config['initialization_kwargs']['length'] = random.randrange(10, 100)
        popsize = random.randrange(50, 500)
        pop = random_pop(popsize, **config['initialization_kwargs'])
        hits = {x:0 for x in pop}
        outsize = popsize + 1
        for individual in sel.uniform_random_selection(pop, outsize):
            hits[individual] += 1
        ones = 0
        greater = 0
        for hit in hits.values():
            if hit == 1:
                ones += 1
            elif hit >= 2:
                greater += 1
        assert ones <= outsize - 1
        assert greater >= 1
    
    def test_probably_duplicates(self):
        #sometimes has duplicates
        popsize = random.randrange(50, 500)
        failures = 0
        outsize = popsize - 1
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            hits = {x:0 for x in pop}
            for individual in sel.uniform_random_selection(pop, outsize):
                hits[individual] += 1
            failed = True
            for hit in hits.values():
                if hit > 1:
                    failed = False
                    break
            if failed:
                failures += 1
        assert failures < max(iterations / 10, 2)

    def test_population_unmodified(self):
        #selection doesn't modify the input population
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            copies = [copy.deepcopy(x) for x in pop]
            selection = sel.uniform_random_selection(pop, random.randint(1, popsize * 2))
            for i in range(popsize):
                assert same_object(pop[i], copies[i])

class TestKTournamentWithReplacement:
    def test_output_size(self):
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize * 2)
            k = random.randint(2, popsize)
            assert len(sel.k_tournament_with_replacement(pop, outsize, k=k)) == outsize
        assert len(sel.k_tournament_with_replacement(pop, outsize, k=popsize)) == outsize
        assert len(sel.k_tournament_with_replacement(pop, outsize, k=2)) == outsize

    def test_elitist(self):
        #never selects any of the worst k-1 individuals
        for _ in range(iterations * 10): #this one is very prone to false passes
            popsize = random.randrange(50, 500)
            outsize = popsize * 3
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            k = random.randint(2, popsize-1)
            worst_individuals = sorted(pop, key=lambda x:x.fitness)[:k-1]
            out = sel.k_tournament_with_replacement(pop, outsize, k=k)
            for ind in worst_individuals:
                assert ind not in out
    
    def test_is_based_on_fitness(self):
        #individuals with a higher fitness are selected more often
        popsize = random.randrange(50, 500)
        outsize = popsize * 100
        out_of_order = 0
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            hits = {x:0 for x in pop}
            k = random.randint(2, popsize-1)
            for individual in sel.k_tournament_with_replacement(pop, outsize, k=k):
                hits[individual] += 1
            ordering = sorted(pop, key=lambda x:x.fitness)
            for i in range(len(ordering) - 1):
                if hits[ordering[i]] > hits[ordering[i + 1]]:
                    out_of_order += 1
        assert out_of_order < popsize * iterations / 10

    def test_definitely_duplicates(self):
        #always has duplicates when out > in
        popsize = random.randrange(50, 500)
        config['initialization_kwargs']['length'] = random.randrange(10, 100)
        pop = random_pop(popsize, **config['initialization_kwargs'])
        random_fitness(pop)
        hits = {x:0 for x in pop}
        outsize = popsize + 1
        for individual in sel.k_tournament_with_replacement(pop, outsize, k=2):
            hits[individual] += 1
        ones = 0
        greater = 0
        for hit in hits.values():
            if hit == 1:
                ones += 1
            elif hit >= 2:
                greater += 1
        assert ones <= outsize - 1
        assert greater >= 1
    
    def test_probably_duplicates(self):
        #sometimes has duplicates
        popsize = random.randrange(50, 500)
        failures = 0
        outsize = popsize - 1
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            hits = {x:0 for x in pop}
            for individual in sel.k_tournament_with_replacement(pop, outsize, k=2):
                hits[individual] += 1
            failed = True
            for hit in hits.values():
                if hit > 1:
                    failed = False
                    break
            if failed:
                failures += 1
        assert failures < max(iterations / 10, 2)
    
#     #k > popsize is invalid
#     def test_fails_on_impossible_k(self):
#         popsize = random.randrange(50, 500)
#         config['initialization_kwargs']['length'] = random.randrange(10, 100)
#         pop = random_pop(popsize, **config['initialization_kwargs'])
#         random_fitness(pop)
#         with pytest.raises(Exception):
#             sel.k_tournament_with_replacement(pop, popsize, k=popsize+1)

    def test_population_unmodified(self):
        #selection doesn't modify the input population
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            copies = [copy.deepcopy(x) for x in pop]
            selection = sel.k_tournament_with_replacement(pop, random.randint(1, popsize * 2), k=random.randint(2, popsize))
            for i in range(popsize):
                assert same_object(pop[i], copies[i])

class TestKTournamentWithoutReplacement:
    def test_output_size(self):
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize - 3)
            k = random.randint(2, popsize - outsize)
            assert len(sel.k_tournament_without_replacement(pop, outsize, k=k)) == outsize

    def test_elitist(self):
        #never selects any of the worst k-1 individuals,
        #always selects at least one of the best popsize - outsize - k + 2 individuals
        #get out a sheet of paper and derive that one yourself :)
        for _ in range(iterations * 5): #this one is very prone to false passes
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize - 3)
            k = random.randint(2, popsize - outsize)
            ordered = sorted(pop, key=lambda x:x.fitness)
            worst_individuals = ordered[:k-1]
            best_individuals = ordered[-(popsize-outsize-k+2):]
            out = sel.k_tournament_without_replacement(pop, outsize, k=k)
            for ind in worst_individuals:
                assert ind not in out
            failed = True
            for ind in best_individuals:
                if ind in out:
                    failed = False
                    break
            assert not failed
    
    def test_is_based_on_fitness(self):
        #individuals with a higher fitness are selected more often
        popsize = random.randrange(50, 500)
        config['initialization_kwargs']['length'] = random.randrange(10, 100)
        pop = random_pop(popsize, **config['initialization_kwargs'])
        outsize = popsize // 2
        k = 2
        random_fitness(pop)
        hits = {x:0 for x in pop}
        out_of_order = 0
        for _ in range(iterations):
            for individual in sel.k_tournament_without_replacement(pop, outsize, k=k):
                hits[individual] += 1
        ordering = sorted(pop, key=lambda x:x.fitness)
        for i in range(len(ordering) - 1):
            if hits[ordering[i]] > hits[ordering[i + 1]]:
                out_of_order += 1
        assert out_of_order < popsize * iterations / 10

    def test_definitely_no_duplicates(self):
        #no duplicates allowed
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize - 3)
            k = random.randint(2, popsize - outsize)
            hits = {x:0 for x in pop}
            for individual in sel.k_tournament_without_replacement(pop, outsize, k=2):
                hits[individual] += 1
            for hit in hits.values():
                assert hit <= 1
    
#     #k > popsize - outsize + 1 is invalid
#     def test_fails_on_impossible_k(self):
#         popsize = random.randrange(50, 500)
#         config['initialization_kwargs']['length'] = random.randrange(10, 100)
#         pop = random_pop(popsize, **config['initialization_kwargs'])
#         random_fitness(pop)
#         outsize = popsize // 2
#         k = (popsize - outsize) + 2
#         with pytest.raises(Exception):
#             sel.k_tournament_without_replacement(pop, outsize, k=k)

    def test_population_unmodified(self):
        #selection doesn't modify the input population
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            copies = [copy.deepcopy(x) for x in pop]
            outsize = random.randint(1, popsize - 3)
            k = random.randint(2, popsize - outsize)
            selection = sel.k_tournament_without_replacement(pop, outsize, k=k)
            for i in range(popsize):
                assert same_object(pop[i], copies[i])

class TestFitnessProportionate:
    def test_output_size(self):
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize * 2)
            assert len(sel.fitness_proportionate_selection(pop, outsize)) == outsize
    
    def test_strongly_prefers_outliers(self):
        #very high fitness means very high bias
        popsize = random.randrange(50, 500)
        outsize = popsize * 3
        min_fit = -10
        max_fit = 10
        big_fit = max_fit + ((max_fit - min_fit) * 1000000)
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop, min_fit, max_fit)
            pop[random.randint(0, len(pop)-1)].fitness = big_fit
            not_it = 0
            for ind in sel.fitness_proportionate_selection(pop, outsize):
                if ind.fitness <= max_fit:
                    not_it += 1
            assert not_it < outsize/10
    
    def test_is_uniform(self):
        #uniform fitness means uniform random selection
        popsize = random.randrange(50, 500)
        out_of_bounds = 0
        outsize = popsize * 1000
        expected = outsize / popsize
        min_bound = max(expected / 1.2, 1)
        max_bound = expected * 1.2
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            standard_fitness(pop)
            hits = {x:0 for x in pop}
            out = sel.fitness_proportionate_selection(pop, outsize)
            for individual in out:
                hits[individual] += 1
            for hit in hits.values():
                if min_bound > hit or max_bound < hit:
                    out_of_bounds += 1
        assert out_of_bounds <= 1.2 * iterations

    def test_definitely_duplicates(self):
        #always has duplicates when out > in
        popsize = random.randrange(50, 500)
        config['initialization_kwargs']['length'] = random.randrange(10, 100)
        pop = random_pop(popsize, **config['initialization_kwargs'])
        random_fitness(pop)
        hits = {x:0 for x in pop}
        outsize = popsize + 1
        for individual in sel.fitness_proportionate_selection(pop, outsize):
            hits[individual] += 1
        ones = 0
        greater = 0
        for hit in hits.values():
            if hit == 1:
                ones += 1
            elif hit >= 2:
                greater += 1
        assert ones <= outsize - 1
        assert greater >= 1
    
    def test_probably_duplicates(self):
        #sometimes has duplicates
        popsize = random.randrange(50, 500)
        failures = 0
        outsize = popsize - 1
        for _ in range(iterations):
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            hits = {x:0 for x in pop}
            for individual in sel.fitness_proportionate_selection(pop, outsize):
                hits[individual] += 1
            failed = True
            for hit in hits.values():
                if hit > 1:
                    failed = False
                    break
            if failed:
                failures += 1
        assert failures < max(iterations / 10, 2)

    def test_population_unmodified(self):
        #selection doesn't modify the input population
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            copies = [copy.deepcopy(x) for x in pop]
            outsize = random.randint(1, popsize * 2)
            selection = sel.fitness_proportionate_selection(pop, outsize)
            for i in range(popsize):
                assert same_object(pop[i], copies[i])

class TestTruncation:
    def test_output_size(self):
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize)
            assert len(sel.truncation(pop, outsize)) == outsize

    def test_no_baddies(self):
        #has none of the worst individuals
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize)
            worst_individuals = sorted(pop, key=lambda x:x.fitness)[:popsize-outsize]
            out = sel.truncation(pop, outsize)
            for ind in worst_individuals:
                assert ind not in out

    def test_all_goodies(self):
        #has all of the best individuals
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize)
            best_individuals = sorted(pop, key=lambda x:x.fitness)[popsize-outsize:]
            out = sel.truncation(pop, outsize)
            for ind in best_individuals:
                assert ind in out

    def test_no_duplicates(self):
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            outsize = random.randint(1, popsize)
            hits = {x:0 for x in pop}
            for individual in sel.truncation(pop, outsize):
                hits[individual] += 1
            for hit in hits.values():
                assert hit <= 1

    def test_population_unmodified(self):
        #selection doesn't modify the input population
        for _ in range(iterations):
            popsize = random.randrange(50, 500)
            config['initialization_kwargs']['length'] = random.randrange(10, 100)
            pop = random_pop(popsize, **config['initialization_kwargs'])
            random_fitness(pop)
            copies = [copy.deepcopy(x) for x in pop]
            outsize = random.randint(1, popsize)
            selection = sel.truncation(pop, outsize)
            for i in range(popsize):
                assert same_object(pop[i], copies[i])