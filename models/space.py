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
        return [(self.center[0],self.center[1],i) for i in self.coords[2][int(self.points[2]/2):]]

    @cached_property
    def list_mesh(self):
        """ A list of all coords """
        return np.array(np.meshgrid(*self.coords)).T.reshape(-1,len(self.points))

    @property
    def list(self):
        """ An iterable of all coords """
        return itertools.product(*[range(p) for p in self.points])

    def rz(self):
        indices = np.indices(self.points)
        z = np.abs(indices[0] - self.center[0])*self.scale
        
        deltas = np.array([(indices[i+1]-c)*self.scale for i, c in enumerate(self.center[1:])])
        r = np.sum(deltas**2, axis=0)**0.5
        return r, z