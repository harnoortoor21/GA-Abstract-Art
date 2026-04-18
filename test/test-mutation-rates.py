import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chromosome import build_chromosome
from src.ga.crossover import crossover
from src.ga.mutation import mutation


features = {
    "energy": 0.9,
    "valence": 0.8,
    "density": 0.7
}


def compare(before, after):
    changes = {}
    for key in before:
        if before[key] != after[key]:
            changes[key] = (before[key], after[key])
    return changes


def test_mutation_100():
    mutation_counts = {
        "scale": 0,
        "warp_strength": 0,
        "octaves": 0,
        "persistence": 0,
        "palette_id": 0
    }

    total_runs = 100

    for i in range(total_runs):


        parent1 = build_chromosome(features)
        parent2 = build_chromosome(features)


        child = crossover(parent1, parent2)


        before = child.copy()


        mutated = mutation(child)


        changes = compare(before, mutated)

        print(f"\n--- RUN {i+1} ---")
        print("BEFORE:", before)
        print("AFTER :", mutated)

        if changes:
            print("CHANGED GENES:")
            for k, v in changes.items():
                print(f"  {k}: {v[0]} → {v[1]}")

                if k in mutation_counts:
                    mutation_counts[k] += 1
        else:
            print("No mutation occurred")

    print("\n========== MUTATION SUMMARY ==========")
    for k, v in mutation_counts.items():
        print(f"{k}: mutated in {v}/{total_runs} runs")


if __name__ == "__main__":
    test_mutation_100()