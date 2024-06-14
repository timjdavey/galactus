import sys

sys.path.append("../")

from models.params import large_points, large_z
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_baselines, generate_variants
from models.workers import VARIANTS


if __name__ == "__main__":
    profiles = quality_profiles(3)
    points, z = large_points, large_z

    baselines = generate_baselines(profiles, points, z)
    generate_variants(profiles, points, z, VARIANTS, baselines)
