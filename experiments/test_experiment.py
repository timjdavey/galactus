import sys
sys.path.append("../")

import pickle
from models.simulation import Simulation
from models.space import Space

def main():
    size = 5
    space = Space([size,size,size], scale=0.1)
    
    masses = space.blank()
    masses[space.center[0]][space.center[1]][int(space.points[2]*0.2)] = 1
    masses[space.center[0]][space.center[1]][int(space.points[2]*0.8)] = 1
    
    solar = Simulation(masses, space)
    solar.analyse(multi=True)
    print(solar.sums)

    with open('tester.pickle', 'wb') as fh:
        pickle.dump(solar, fh)

if __name__ == '__main__':
    main()