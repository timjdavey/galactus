import sys
sys.path.append("../")

from references.milkyway import profiles
from models.sersic.profile import SersicProfile
from models.sersic.galaxy import generate_galaxy

if __name__ == '__main__':
    pname = 'mcmillian2011best'
    
    space_points = 800
    radius = 40

    sim = SersicProfile(profiles[pname]).galaxy(
        space_points=space_points,
        calc_points=0,
        radius=radius,
        cp=print)

    # general points
    point_sections = [
        # points, upto radius
        (15, 5), 
        (20, 25),
        (10, radius),
    ]
    for calc_points, calc_radius in point_sections:
        sim.analyse(sim.radius_points(calc_radius, calc_points))
    
    # earth interp
    space = sim.space
    solar = 8.3
    left = len(space.x[sim.space.x<8.3])
    sim.analyse([
        (space.center[0], space.center[1], left),
        (space.center[0], space.center[1], left+1)
    ])

    # save
    filename = "_".join([str(s) for s in [pname, space_points, radius]])
    sim.save(filename, masses=False)