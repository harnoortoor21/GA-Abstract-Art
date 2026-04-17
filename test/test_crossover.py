import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chromosome import build_chromosome
from src.ga.crossover import crossover

features = {
    "energy": 0.9,
    "valence": 0.7,
    "density": 0.9
}


def diff(parent1, parent2, child):
    """
    Compare how far child is from parents
    """
    results = {}

    for key in parent1:
        if isinstance(parent1[key], (int, float)):

            avg_parent = (parent1[key] + parent2[key]) / 2

            results[key] = {
                "p1": parent1[key],
                "p2": parent2[key],
                "child": child[key],
                "distance_from_mid": abs(child[key] - avg_parent)
            }

        else:
            # categorical (palette_id etc.)
            results[key] = {
                "p1": parent1[key],
                "p2": parent2[key],
                "child": child[key],
                "is_from_parent": child[key] in [parent1[key], parent2[key]]
            }

    return results


def test_crossover_100():
    stats = {
        "numeric_mid_distance": [],
        "gene_inheritance_rate": 0,
        "runs": 100
    }

    for i in range(100):

        parent1 = build_chromosome(features)
        parent2 = build_chromosome(features)

        child = crossover(parent1, parent2)

        comparison = diff(parent1, parent2, child)

        print(f"\n--- RUN {i+1} ---")

        run_mid_distances = []

        for k, v in comparison.items():

            print(f"{k}: {v}")

            # numeric genes
            if "distance_from_mid" in v:
                run_mid_distances.append(v["distance_from_mid"])

        if run_mid_distances:
            stats["numeric_mid_distance"].append(np.mean(run_mid_distances))

    print("\n========== CROSSOVER SUMMARY ==========")

    print(f"Avg distance from midpoint: {np.mean(stats['numeric_mid_distance']):.4f}")
    print(f"Std deviation: {np.std(stats['numeric_mid_distance']):.4f}")

    print("\nInterpretation:")
    print("- low value → child is true blend (strong crossover)")
    print("- high value → child is more random / mutation-like")


if __name__ == "__main__":
    test_crossover_100()