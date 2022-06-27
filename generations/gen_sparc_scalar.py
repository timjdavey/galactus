import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles


if __name__ == '__main__':

    for k, p in quality_profiles().items():
        smap = p.generate_maps(load=True, DIR='')
        #sim = smap.generate_scalar_galaxy()
        #sim.save("sparc_scalar_%s" % k, masses=False)

        sim = smap.generate_scalar_galaxy(potential=True, alpha=0.25)
        sim.save("sparc_potential_25_%s" % k, masses=False)
        print(k)