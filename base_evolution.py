class BaseEvolutionPopulation():
    def __init__(self, individual_class, mu, num_children, mutation_rate,
                 parent_selection, survival_selection,
                 initialization_kwargs=dict(), parent_selection_kwargs=dict(),
                 recombination_kwargs = dict(), mutation_kwargs = dict(),
                 survival_selection_kwargs=dict(), **kwargs):
        self.mu = mu
        self.num_children = num_children
        self.mutation_rate = mutation_rate
        self.parent_selection = parent_selection
        self.survival_selection = survival_selection
        self.parent_selection_kwargs = parent_selection_kwargs
        self.recombination_kwargs = recombination_kwargs
        self.mutation_kwargs = mutation_kwargs
        self.survival_selection_kwargs = survival_selection_kwargs

        self.population = individual_class.initialization(self.mu, **initialization_kwargs)

    def generate_children(self):
        children = list()

        # TODO: Select parents
        # hint: self.parent_selection(self.population, **self.parent_selection_kwargs)
        pass

        # TODO: Recombine parents to generate children
        pass

        # TODO: Mutate children if appropriate
        pass

        # return children
        return children


    def survival(self):
        self.population = self.survival_selection(self.population, self.mu, **self.survival_selection_kwargs)
