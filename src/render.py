"""
renderer.py

This module is responsible for converting a chromosome into a visual image.
It uses the OpenCV library to draw shapes onto a canvas based on the
parameters stored in the chromosome.

Each shape's position, size, color, and opacity are used to render
an abstract image.

The output is a generated image that visually represents a chromosome,
which can then be evaluated by the fitness function or saved to disk.
"""

import numpy as np
import cv2
import os
from opensimplex import noise2
from datetime import datetime

# BGR
# we could have more colours in the palette, but if we add more right now it will use the same colours (all colours) every time for the happy/sad image
# idk how to make it so that it uses different colours from the palette each time, maybe we can have a random seed for the palette and then use that seed to shuffle the palette each time?

PALETTE_HAPPY = [
    (10, 10, 40), # dark red
    (30, 20, 80),  # Red
    (60, 40, 120),  # Light Red
    (100, 80, 160),  # Lighter Red
    (160, 140, 200)  # Very Light Red
]

PALETTE_SAD = [
    (94, 4, 3),    # Dark Blue
    (182, 119, 0),  # Blue
    (216, 180, 0), # Light Blue
    (239, 224, 144), # Lighter Blue
]


# smooth colour blending between colours in a palette based on a value t between 0 and 1
def interpolate_colour(t, palette):
    # Interpolate between colors in the palette based on t.
    scaled = t * (len(palette) - 1)
    #floors the value to get the index of the first color
    idx = int(scaled)
    idx = min(idx, len(palette) - 2)  # ensure idx is within bounds for palette access
    # how far between the two colors we are, 0.0 meaning we are exactly on the first color
    # 1.0 meaning we are exactly on the second color. 0.4 would mean we are 40% of the way
    # from the first color to the second color.
    frac = scaled - idx
    # gets the 2 colours as BGR values and makes numpy array
    c1 = np.array(palette[idx], dtype=np.float32)
    c2 = np.array(palette[idx + 1], dtype=np.float32)
    # linear interpolation (lerp) between the two colors, channel by channel
    # if frac is 0.0, we get c1, if frac is 1.0, we get c2, at 0.5 we get an even mix of both
    return tuple((c1 * (1 - frac) + c2 * frac).astype(np.uint8))
    # the astype is to convert the result back to uint8 which is the format for colours in OpenCV


# second layer of blending of 2 colours from 2 different palettes, based on a blend value between 0 and 1,
# where 0 means we get the colour from palette_a and 1 means we get the colour from palette_b,
# and values in between give us a mix of both colours

def blend_palettes(t, palette_a, palette_b, blend):
    # get the colour from each palette based on t, which is the same for both palettes so we get corresponding colours
    ca = np.array(interpolate_colour(t, palette_a), dtype=np.float32)
    cb = np.array(interpolate_colour(t, palette_b), dtype=np.float32)
    # blend the two colours based on the blend value, which is a simple linear interpolation (lerp) between the two colours and convert bak to RGB
    return tuple((ca * (1 - blend) + cb * blend).astype(np.uint8))

# t moves along one palette, blend moves between 2 palettes
# right now it uses all colours from both palettes, and blend changes depending on valence, so v = 0 would be all colours from the sad palette,
# v = 1 would be all colours from the happy palette, and v = 0.5 would be an even mix of both palettes
# t value picks color from 1-5, and how much of each colour from one palette (does this once for each palette), then blend picks how much of the colour from each palette we use

def make_fluid_image(width, height, chromosome):
    # Create a blank canvas as a 3D numpy array (height x width x 3 for RGB or is it BGR?), every value set to 0 (black to start)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    #set values based on chromosome
    scale = chromosome["scale"]
    warp_strength = chromosome["warp_strength"]
    seed_x        = chromosome["seed_x"]
    seed_y        = chromosome["seed_y"]
    flow_angle    = chromosome["flow_angle"]
    octaves       = chromosome["octaves"]
    persistence   = chromosome["persistence"]
    valence       = chromosome["valence"]   # drives palette blend, will not be in chromosome
    dx = np.cos(flow_angle)
    dy = np.sin(flow_angle)

    print("Generating image...")

    for y in range(height):
        if y % (height // 10) == 0:
            print(f"Progress: {int(y / height * 100)}%")

        for x in range(width):

            # warp vectors with octaves
            nx, ny = 0.0, 0.0
            amp, freq = 1.0, 1.0
            for _ in range(octaves):
                nx += amp * noise2((x + seed_x) * freq / scale, (y + seed_y) * freq / scale)
                ny += amp * noise2((x + seed_x + 300) * freq / scale, (y + seed_y + 300) * freq / scale)
                amp  *= persistence
                freq *= 2.0

            warped_x = x + warp_strength * nx * dx
            warped_y = y + warp_strength * ny * dy

            # final value with octaves
            val, amp, freq = 0.0, 1.0, 1.0
            for _ in range(octaves):
                val  += amp * noise2(warped_x * freq / scale, warped_y * freq / scale)
                amp  *= persistence
                freq *= 2.0

            t = (val + 1) / 2
            t = max(0.0, min(1.0, t))

            canvas[y, x] = blend_palettes(t, PALETTE_SAD, PALETTE_HAPPY, valence)

    return canvas


# irrelevant
# test chromosome
def build_chromosome_from_midi(midi_features: dict) -> dict:
    """
    Convert MIDI features into a starting chromosome.
    The GA will evolve from this point rather than pure random.
    """
    energy    = midi_features["energy"]
    valence   = midi_features["valence"]
    density   = midi_features["density"]

    return {
        # high energy = tight chaotic shapes, low energy = slow sweeping
        "scale":        np.random.uniform(40, 120) * (1.0 - energy * 0.5),

        # high energy = more distortion
        "warp_strength": 50 + energy * 130,

        # high density = more octave detail
        "octaves":      max(1, round(1 + density * 5)),

        # high energy = rougher detail
        "persistence":  0.3 + energy * 0.4,

        # valence drives palette (0 = sad/cool, 1 = happy/warm)
        "valence":      valence,

        "seed_x":       np.random.uniform(0, 10000),
        "seed_y":       np.random.uniform(0, 10000),
        "flow_angle":   np.random.uniform(0, 2 * np.pi),
    }


# MAIN TEST

# placeholder — replace with actual midi extractor output to test
midi_features = {
    "energy":  0.7, # high energy = faster song
    "valence": 0.4, # ranges from 0-1, where 0 is sad/cool and 1 is happy/warm
    "density": 0.6, # notes per second, higher means faster, detailed image, lower means slower, simpler image
}

chromosome = build_chromosome_from_midi(midi_features)

print("Chromosome parameters:")
for key, value in chromosome.items():
    print(f"  {key}: {value}")


# image outputting and saving
img = make_fluid_image(512, 512, chromosome)

folder = "fluid-v1 outputs"
os.makedirs(folder, exist_ok=True)
timestamp = datetime.now().strftime("%d%H%M%S")
filename  = os.path.join(folder, f"fluid-v1_{timestamp}.png")

cv2.imshow("fluid", img)
cv2.imwrite(filename, img)
cv2.waitKey(0)