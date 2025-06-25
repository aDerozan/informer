import numpy as np

class VultureOptimizer:
    def __init__(self, objective_function, bounds, max_iter=100, population_size=50):
        self.objective_function = objective_function
        self.bounds = bounds
        self.max_iter = max_iter
        self.population_size = population_size

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (lower, upper) in self.bounds.items():
                individual[param] = np.random.uniform(lower, upper)
            population.append(individual)
        return population

    def evaluate_population(self, population):
        fitness_values = []
        for individual in population:
            fitness = self.objective_function(individual)
            fitness_values.append(fitness)
        return fitness_values

    def optimize(self):
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.max_iter):
            fitness_values = self.evaluate_population(population)

            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness_values[best_idx]

            for i in range(self.population_size):
                population[i] = self._update_individual(population[i], best_solution)

        return best_solution

    def _update_individual(self, individual, best_solution):
        new_individual = {}
        for param in individual:
            new_individual[param] = np.random.uniform(self.bounds[param][0], self.bounds[param][1])
        return new_individual
