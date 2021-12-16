import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Space:
    def __init__(self, masses, distances=None, G=1, cp=None):
        """
        Basic 
        """
        points = masses[0].shape
        self.points = points
        self.masses = np.array(masses)
        self.G = G
        self.cp = cp
        
        for i, mass in enumerate(self.masses):
            if mass.shape != self.points:
                raise ValueError("mass %s shape does not match grid (%s, %s)" % (i, mass.shape, self.points))

        if distances is None:
            distances = points
        self.distances = distances
        
        self.coord_x = np.linspace(0, distances[1], points[1])
        self.coord_y = np.linspace(0, distances[0], points[0])
        self.space_x, self.space_y = np.meshgrid(self.coord_x, self.coord_y)
        self._fields = None
    
    @property
    def ratio(self):
        """ Ratio of y/x points"""
        return self.points[0]/self.points[1]
    
    @property
    def fields(self):
        """ dict of field calculations """
        if self._fields is None:
            self.calc()
        return self._fields
        
    def calc(self):
        """ Calculates the gravitational & mass fields for the space """
        
        # first assumption is all masses
        # in each grid point
        # are summed to a single mass for that grid
        # so densities are treated normally

        mass = np.sum(self.masses, axis=0)
        blank = np.zeros(self.points)
        

        # then calculates the affect of the mass in
        # each grid point on all other grid points

        # we approximate the mass within a given grid
        # to be the center of mass for the grid
        # but outside of each grid, we never make that assumption

        gridxs = []
        gridys = []
        gridabs = []
        total = len(self.coord_x)
        for ix, x in enumerate(self.coord_x):
            if self.cp is not None: self.cp(total-ix)
            for iy, y in enumerate(self.coord_y):
                
                
                gx, gy, ab = blank.copy(), blank.copy(), blank.copy()
                M = mass[iy][ix]
                
                for iu, u in enumerate(self.coord_x):
                    for iv, v in enumerate(self.coord_y):

                        # the mass of a certain grid point
                        # has no impact on it's own grid point
                        # to avoid infinities etc

                        if x == u and y == v:
                            pass
                        else:
                            dx, dy = u-x, v-y
                            r2 = (dx**2+dy**2)
                            r = r2**0.5
                            F = self.G*M/r2
                            gx[iv][iu] = -F*(dx/r)
                            gy[iv][iu] = -F*(dy/r)
                            ab[iv][iu] = np.abs(F)
                
                gridxs.append(gx)
                gridys.append(gy)
                gridabs.append(ab)

        if self.cp is not None: self.cp('')
                
        self._fields = {
            'mass': mass,
            'gravities_x': gridxs,
            'gravities_y': gridys,
            'gx': np.sum(gridxs, axis=0),
            'gy': np.sum(gridys, axis=0),
            'abs': np.sum(gridabs, axis=0),
        }
        return self._fields
    
    def quiver(self, size=15, scale=20):
        """ Plots quiver plot of 2D Space"""
        fields = self.fields
        plt.figure(figsize=(size,size*self.ratio))
        plt.quiver(self.space_x, self.space_y, fields['gx'], fields['gy'], fields['abs'], scale=scale)
        plt.show()
    
    def line(self, position=None, ylim=(None, None)):
        """ Plots a specific line across the 2D Space """
        if position is None: position = int(len(self.coord_y)/2)
        
        plt.figure(figsize=(20,10))
        
        data = {}
        #data['Standard gravity'] = self.fields['gx'][position]
        data['Absolute standard'] = np.abs(self.fields['gx'][position])
        data['Tau'] = self.fields['abs'][position]
        external = data['Tau'] - data['Absolute standard']
        data['Adjusted'] = data['Absolute standard']/(1+external/data['Absolute standard'])
        
        for k,v in data.items():
            g = sns.lineplot(x=self.coord_x, y=v, label=k)
        
        g.set(ylim=ylim)
        g.axhline(0, ls='--', color='lightgrey')

