import sys

sys.path.append("../")

from models.params import large_points, large_z
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_baselines, generate_variants
from models.workers import VARIANTS


if __name__ == "__main__":
    profiles = quality_profiles(1)
    z = 11
    for points in (61, 65, 75, 81):
        baselines = generate_baselines(profiles, points, z)
        generate_variants(profiles, points, z, VARIANTS, baselines)
