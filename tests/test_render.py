from src.chromosome import build_chromosome
from src.render import make_fluid_image, save_image, show_image

test_features = {
    'energy': 0.667,
    'valence': 0.589,
    'density': 0.344
}

chromosome = build_chromosome(test_features)

chromosome["palette_id"] = 21

print("Chromosome:")
for key, value in chromosome.items():
    print(f"{key}: {value}")

img = make_fluid_image(512, 512, chromosome)
show_image(img)
save_image(img)