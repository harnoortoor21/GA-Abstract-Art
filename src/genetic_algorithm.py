"""
genetic_algorithm.py

This module implements the genetic algorithm used to evolve abstract images.
It generates an initial population of chromosomes (image representations)
and iteratively improves them over multiple generations.

Key operations include:
- Selection (binary tournament)
- Crossover (combining parent chromosomes)
- Mutation (random variation)
- Replacement (maintaining population diversity)

The algorithm uses the fitness function to evaluate how well each
chromosome matches the emotional features of the input music.

The output is the best-performing chromosome representing the final image.
"""

from src.chromosome import build_chromosome
import random
from src.fitness import calculate_fitness
from src.ga.crossover import crossover
from src.ga.mutation import mutation

POPULATION_SIZE = 20
GENERATIONS = 50
TOURNAMENT_K = 3


def create_population(n):
    """Create initial population of random chromosomes"""
    population = []
    for i in range(n):
        population.append(build_chromosome())
    return population


def tournament_selection(population, target_features, k=TOURNAMENT_K):
    """
    Select best individual from random k individuals
    """
    group = []
    for i in range(k):
        index = random.randint(0, len(population) - 1)
        group.append(population[index])

    best = max(group, key=lambda ind: calculate_fitness(ind, target_features))
    return best


def find_best(population, target_features):
    """Find best chromosome in population"""
    return max(population, key=lambda ind: calculate_fitness(ind, target_features))


def ga_main(target_features):
    """
    Main genetic algorithm loop
    """
    population = create_population(POPULATION_SIZE)

    best_ever = find_best(population, target_features)

    for generation in range(GENERATIONS):
        new_population = []

        # Elitism: keep best individual
        best = find_best(population, target_features)
        new_population.append(best)

        # Generate rest of population
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, target_features, TOURNAMENT_K)
            p2 = tournament_selection(population, target_features, TOURNAMENT_K)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)

        # Random replacement for diversity (10% chance)
        if random.random() < 0.1 and len(new_population) > 1:
            replace_index = random.randint(1, len(new_population) - 1)  # Don't replace elite
            new_population[replace_index] = build_chromosome()

        # Track best overall
        new_best = find_best(new_population, target_features)
        if calculate_fitness(new_best, target_features) > calculate_fitness(best_ever, target_features):
            best_ever = new_best

        population = new_population

        # Optional: print progress
        print(
            f"Generation {generation + 1}/{GENERATIONS} - Best fitness: {calculate_fitness(best_ever, target_features):.4f}")

    return best_ever
