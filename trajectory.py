"""
Class that creates batches of trajectories for training or visualization.

Author: Jesse Livezey
"""
import numpy as np

_methods = ['acceln']
_default_method = 'acceln'
class Trajectory():
    def __init__(self, N, batch_size, dim, space_size, object_rad, valid=True, 
                 method=None, rng=None, dt = .5):
        """ Create trajectory of points in space iterator.

        Parameters
        ----------
        N : int
            Number of iteratoins to return.
        batch_size : int
            Number of trajectories to return at once.
        dim : int
            Dimensionality of space.
        space_size : list of ints
            Extent of each dimension.
        object_rad : float or int
            Size of objects.
        valid : boolean
            If true, objects are fully contained in space.
        method : string
            Determines how trajectory updates are found.
        rng : RandomState
            Randon generator for trajectories.
        """
        self.N = N
        self.batch_size = batch_size
        self.dim = dim
        self.space_size = space_size
        self.dt = dt

        assert object_rad < space_size
        self.object_rad = object_rad

        self.valid = valid

        if method is None:
            method = _default_method
        assert method in _methods
        self.method = method

        if rng = None:
            self.rng = np.random.RandomState(08092014)
        else:
            assert isinstance(rng, np.random.mtrand.RandomState)
            self.rng = rng
        
        self.current = 0
        self.state = np.zeros(batch_size, dim)

    def __iter__(self):
        return self

    def next(self):
        if self.current > self.N:
            raise StopIteration
        else:
            self.current += 1
            return self.update()
    
    def update():
        if self.method == 'acceln':
            if current == 0:
                self.vel = np.zeros_like(self.state)
                    for d, s, r in zip(xrange(self.dim), self.space_size, self.object_rad):
                        if self.valid:
                            self.state[...,d] = self.rng.uniform(low=r, high=s-r, size=self.batch_size)
                        else:
                            self.state[...,d] = self.rng.uniform(low=0., high=s, size=self.batch_size)
            else:
                self.vel += rng.normal(size=self.vel.shape)
                self.state += self.dt*self.vel
                for d, s, r in zip(xrange(self.dim), self.space_size, self.object_rad):
                    if self.valid:
                        self.state[...,d] = np.clip(self.state[:,d], r, s-r)
                    else:
                        self.state[...,d] = np.clip(self.state[:,d], 0, s)
        else:
            raise NotImplementedError(str(self.method)+' is not implemented '
                                      +'for generating trajectories')
        return self.state()

