import random

class LinearGenotype():
    def __init__(self):
        self.fitness = None
        self.gene = None

    def random_initialization(self, length, x_bounds, y_bounds):
        # TODO: Add random initialization of fixed-length linear gene
        pass

    def recombine(self, mate, method, **kwargs):
        child = LinearGenotype()
        
        # TODO: Recombine genes of self with mate and assign to child's gene member variable
        assert method.casefold() in {'uniform', '1-point crossover', 'bonus'}
        if method.casefold() == 'uniform':
            # perform uniform recombination
            pass
        elif method.casefold() == '1-point crossover':
            # perform 1-point crossover
            pass
        elif method.casefold() == 'bonus':
            ''' 
            This is a red deliverable (i.e., bonus for anyone).

            Implement the bonus crossover operator as described in deliverable
            Red 1 of Assignment 1b.
            '''
            pass

        return child

    def mutate(self, x_bounds, y_bounds, bonus=None, **kwargs):
        copy = LinearGenotype()
        copy.gene = self.gene.copy()
        
        if bonus is None:
            # TODO: mutate gene of copy
            pass
        else:
            ''' 
            This is a red deliverable (i.e., bonus for anyone).

            Implement the bonus crossover operator as described in deliverable
            Red 1 of Assignment 1b.
            '''
            pass

        return copy

    @classmethod
    def initialization(cls, mu, *args, **kwargs):
        population = [cls() for _ in range(mu)]
        for i in range(len(population)):
            population[i].random_initialization(*args, **kwargs)
        return population