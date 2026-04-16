"""
fitness.py

This module evaluates how well a generated image matches the emotional
characteristics of the input music.

It assigns a fitness score to each chromosome based on:
1. Spatial energy: measures visual complexity relative to music energy
2. Colour-mood alignment: compares image colors to expected emotional tones

The combined fitness score is used by the genetic algorithm to guide
the evolution process toward better image representations.

Higher fitness scores indicate a better match between music and image.
"""

def check_fitness(chromosome, target):
# get the features from chromosome
  energy, density, valence = get_features(chromosome)

# calculate the difference between scores from chromosome and target value(based on midi file), taking absolute value and subtracting
# from 1 so that higher the difference between chromosome and target , lower the fitness of the chromosome
  e_score = 1- abs(energy - target["energy"])
  d_score = 1- abs(density - target["density"])
  v_score = 1- abs(valence - target["valence"])

  avg = (e_score, d_score, v_score) / 3
  
# using exponential in fitness to check if fitness increases and reaches closer to target faster 
  fitness = avg**2
  return fitness
