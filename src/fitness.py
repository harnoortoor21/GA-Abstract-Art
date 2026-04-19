FITNESS_WEIGHTS = {
   'palette': 0.4,   # Changed from 0.35
   'scale': 0.2,    # Changed from 0.25
   'warp': 0.2,     # Changed from 0.25
   'octave': 0.2    # Changed from 0.15
}



def get_palette_range(valence, energy):
    """Get palette range based on valence and energy (matches pick_palette logic)"""
    if valence < 0.4:
        return (1, 10)
    else:
        if energy < 0.7:
            return (11, 19)
        else:
            return (16, 25)


def get_ideal_palette(valence, energy):
    """Calculate ideal palette ID based on valence and energy"""
    low, high = get_palette_range(valence, energy)
    mid = (low + high) / 2
    return mid


def calculate_fitness(chromosome, target_features):
    # SCALE
    tgt_energy = target_features["energy"]

    scale_start = 200 - tgt_energy * 125
    scale_end = 300 + tgt_energy * 220

    ideal_s = (scale_start + scale_end) / 2
    dev_s = abs(ideal_s - chromosome["scale"])
    max_error_s = max(abs(ideal_s - scale_start), abs(ideal_s - scale_end))

    scale_score = max(0, 1 - dev_s / max_error_s)

    # PALETTE (now uses both valence and energy)
    tgt_valence = target_features['valence']
    ideal_p = get_ideal_palette(tgt_valence, tgt_energy)
    low_p, high_p = get_palette_range(tgt_valence, tgt_energy)

    dev_p = abs(ideal_p - chromosome["palette_id"])
    max_error_p = max(abs(ideal_p - low_p), abs(ideal_p - high_p))

    palette_score = max(0, 1 - dev_p / max_error_p)

    # OCTAVE
    tgt_density = target_features['density']
    ideal_o = 1 + (tgt_density * 5)
    dev_o = abs(ideal_o - chromosome["octaves"])
    max_error_o = max(abs(ideal_o - 1), abs(ideal_o - 6))

    octave_score = max(0, 1 - dev_o / max_error_o)

    # WARP
    warp_start = 20 + tgt_energy * 130
    warp_end = 50 + tgt_energy * 130

    ideal_w = (warp_start + warp_end) / 2
    dev_w = abs(ideal_w - chromosome["warp_strength"])
    max_error_w = max(abs(ideal_w - warp_start), abs(ideal_w - warp_end))

    warp_score = max(0, 1 - dev_w / max_error_w)

    # COMBINED FITNESS
    fitness_score = (
            palette_score * FITNESS_WEIGHTS['palette'] +
            scale_score * FITNESS_WEIGHTS['scale'] +
            warp_score * FITNESS_WEIGHTS['warp'] +
            octave_score * FITNESS_WEIGHTS['octave']
    )

    return fitness_score