import sys

sys.path.append("../")

from models.params import flat_points
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_baselines, generate_variants
from models.workers import VARIANTS


if __name__ == "__main__":
    profiles = quality_profiles(1)
    points, z = flat_points, 1
    baselines = generate_baselines(profiles, points, z)
    generate_variants(profiles, points, z, VARIANTS, baselines)
