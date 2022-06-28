import numpy as np
import itertools
from functools import cached_property

class Space:
    """
    Simple class for handling the common features of
    coordinate systems etc, for N dimensions.
    """
    def __init__(self, points, scale=1):
        self.points = points
        self.scale = scale

    @property
    def is_even(self):
        return self.points[0] % 2 == 0 or self.points[1] % 2 == 0

    def blank(self):
        """ An empty array of dimensions of space """
        return np.zeros(self.points)#, dtype=np.dtype('float32'))

    @cached_property
    def count(self):
        """ Total number of points in space """
        return np.product(self.points)
    
    @cached_property
    def dimensions(self):
        """ Lists through the dimensions e.g. 0,1,2 for 3d space """
        return range(len(self.points))

    @cached_property
    def center(self):
        """ The center points of space """
        return np.array([int(p/2) for p in self.points])

    @cached_property
    def coords(self):
        """ List of coords for each dimension, of points of the array
        Use *space.scale for distances.
        """
        return [np.array(range(self.points[d])) for d in self.dimensions]

    @cached_property
    def x(self):
        """ The distances of x-axis for use with graphs mostly """
        return (self.coords[-1]-self.center[2])*self.scale

    @cached_property
    def radius_list(self):
        """ A list of all coords on the radius of a galaxy """
        return [(self.center[0],self.center[1],i) for i in self.coords[2][self.points[2]//2:]]

    def list_mesh(self, coords=None):
        """ A list of all coords """
        if coords is None: coords = self.coords
        return np.array(np.meshgrid(*coords)).T.reshape(-1,len(self.points))

    @cached_property
    def symmetric_points(self):
        xy = list(itertools.combinations_with_replacement(range(self.center[1]+1),2))
        arr = []
        for z in range(self.center[0]+1):
            for p in xy:
                arr.append(tuple([z, p[0], p[1]]))
        return arr

    @property
    def list(self):
        """ An iterable of all coords """
        return itertools.product(*[range(p) for p in self.points])

    @property
    def radius(self):
        return self.points[1]*self.scale/2
    
    def rz(self, offset=0.0):
        """
        Returns the matrixes for R & Z, to use as mass indicies

        :offset: allows you to make sure the masses obtained 
        for n to n+1 aren't set at the density of n,
        instead you can offset so they are set at the density of
        n+offset (e.g. half way between n and n+1).
        """
        indices = np.indices(self.points)
        z = np.abs(indices[0] - self.center[0])*self.scale
        
        deltas = np.array([(indices[i+1]-c)*self.scale for i, c in enumerate(self.center[1:])])
        r = np.sum(deltas**2, axis=0)**0.5
        r += offset*self.scale
        return r, z