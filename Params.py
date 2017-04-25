# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:36:59 2015

@author: rrowekamp
"""

from numpy import fromfile,prod,sqrt,ndarray

class Params(object):
    """
    Params is an object that holds multiply numpy arrays representing
    paramters of a model. The parameters can be mathematically manipulated as
    a whole, which is useful for gradient descent. The class also has support
    for copying itself and saving or loading parameters to or from files.
    Constructor inputs:
        params: Can be a file object, a file name, a Params object, a numpy
            array, or a list or tuple of numpy arrays.
        shapes: If params is a file, file name, or single numpy array, shapes
            is a list of tuples that determine the shape of each parameter.
        dtype: If params is a file or file name, the file is read using this
            dtype. Otherwise, the parameters are converted to this dtype.
    WARNING: When using operators, place the Params object on the left of a
        numpy array. Putting it on the left will result in a numpy array rather
        than a Params object.
    """
    def __init__(self,
                 params,
                 shapes=None,
                 dtype=None
                 ):

        # If params is an open file, load parameters from it.
        if isinstance(params,file):
            # shapes is needed to structure loaded values
            assert shapes is not None
            if dtype is None:
                dtype = float
            self.params = [fromfile(params,count=prod(s),dtype=dtype).reshape(s) for s in shapes]

        # If params is a string, try opening that file and loading params from it
        elif isinstance(params,str):
            # shapes is needed to structure loaded values
            assert shapes is not None
            if dtype is None:
                dtype = float
            with open(params,'r') as f:
                self.params = [fromfile(f,count=prod(s),dtype=dtype).reshape(s) for s in shapes]

        # Copy values from Params object, reshaping and changing dtype if necessary
        elif isinstance(params,Params):
            if dtype is None:
                self.params = [p for p in params.getParams()]
            else:
                self.params = [p.astype(dtype) for p in params.getParams()]
            if shapes is not None:
                for p,s in zip(self.params,shapes):
                    p.shape = s

        elif isinstance(params,ndarray):
            assert params.ndim == 1
            # If params is an array of arrays
            if isinstance(params[0],ndarray):
                if dtype is None:
                    self.params = [p.copy() for p in params]
                else:
                    self.params = [p.astype(dtype) for p in params]

            # if params is a vector of values
            else:
                assert shapes is not None
                size = [prod(s) for s in shapes]
                size.insert(0,0)
                assert params.size == sum(size)
                ind = [sum(size[:j+1]) for j in range(len(size))]
                if dtype is None:
                    self.params = [params[ind[j]:ind[j+1]].reshape(shapes[j]) for j in range(len(size)-1)]
                else:
                    self.params = [params[ind[j]:ind[j+1]].reshape(shapes[j]).astype(dtype) for j in range(len(size)-1)]

        # If params is a list/tuple of arrays
        else:
            assert isinstance(params,tuple) or isinstance(params,list)
            if dtype is None:
                self.params = [p.copy() for p in params]
            else:
                self.params = [p.astype(dtype) for p in params]

    # Returns a list with copies of the parameters
    def getParams(self):

        return [p.copy() for p in self.params]

    # Returns a Params object with copies of the parameters
    def copy(self):

        return Params(self.getParams())

    # Left addition
    def __add__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()

            return Params([AA+BB for AA,BB in zip(p1,p2)])
        else:
            return Params([AA+A for AA in self.params])

    # Right addition
    def __radd__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()

            return Params([BB+AA for AA,BB in zip(p1,p2)])
        else:
            return Params([A+AA for AA in self.params])

    # Left subtraction
    def __sub__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()

            return Params([AA-BB for AA,BB in zip(p1,p2)])
        else:
            return Params([AA-A for AA in self.params])

    # Right subtraction
    def __rsub__(self,A):

        if isinstance(A,Params):


            p1 = self.params
            p2 = A.getParams()

            return Params([BB-AA for AA,BB in zip(p1,p2)])
        else:
            return Params([A-AA for AA in self.params])

    # Left multiplication
    def __mul__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()
            return Params([AA*BB for AA,BB in zip(p1,p2)])
        else:
            return Params([AA*A for AA in self.params])

    # Right multiplication
    def __rmul__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()
            return Params([BB*AA for AA,BB in zip(p1,p2)])
        else:
            return Params([A*AA for AA in self.params])

    # Left floor division
    def __floordiv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            return Params([AA//BB for AA,BB in zip(p1,p2)])
        else:
            return Params([AA//A for AA in self.params])

    # Left division
    def __div__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            return Params([AA/BB for AA,BB in zip(p1,p2)])
        else:
            return Params([AA/A for AA in self.params])

    # Left true division
    def __truediv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            return Params([AA/BB for AA,BB in zip(p1,p2)])
        else:
            return Params([AA/A for AA in self.params])

    # Right floor division
    def __rfloordiv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            return Params([BB//AA for AA,BB in zip(p1,p2)])
        else:
            return Params([A//AA for AA in self.params])

    # Right division
    def __rdiv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            return Params([BB/AA for AA,BB in zip(p1,p2)])
        else:
            return Params([A/AA for AA in self.params])

    # Right true division
    def __rtruediv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            return Params([BB/AA for AA,BB in zip(p1,p2)])
        else:
            return Params([A/AA for AA in self.params])

    # Left power
    def __pow__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()

            return Params([AA**BB for AA,BB in zip(p1,p2)])
        else:

            return Params([AA**A for AA in self.params])

    # Right power
    def __rpow__(self,A):

        if isinstance(A,Params):

            p1 = self.params
            p2 = A.getParams()

            return Params([BB**AA for AA,BB in zip(p1,p2)])
        else:

            return Params([A**AA for AA in self.params])

    # In-place addition
    def __iadd__(self,A):

        if isinstance(A,Params):

            p2 = A.getParams()

            for AA,BB in zip(self.params,p2):
                AA += BB

            return self
        else:

            for AA in self.params:
                AA += A

            return self

    # In-place subtraction
    def __isub__(self,A):

        if isinstance(A,Params):

            p2 = A.getParams()

            for AA,BB in zip(self.params,p2):
                AA -= BB

            return self
        else:

            for AA in self.params:
                AA -= A

            return self

    # In-place multiplication
    def __imul__(self,A):

        if isinstance(A,Params):

            p2 = A.getParams()

            for AA,BB in zip(self.params,p2):
                AA *= BB

            return self
        else:

            for AA in self.params:
                AA *= A

            return self

    # In-place floor division
    def __ifloordiv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            for AA,BB in zip(p1,p2):
                AA //= BB

            return self
        else:

            for AA in self.params:
                AA //= A

            return self

    # In-place division
    def __idiv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            for AA,BB in zip(p1,p2):
                AA /= BB

            return self
        else:

            for AA in self.params:
                AA /= A

            return self

    # In-place true division
    def __itruediv__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            for AA,BB in zip(p1,p2):
                AA /= BB

            return self
        else:

            for AA in self.params:
                AA /= A

            return self

    # In-place power
    def __ipow__(self,A):

        if isinstance(A,Params):
            p1 = self.params
            p2 = A.getParams()

            for AA,BB in zip(p1,p2):
                AA **= BB

            return self
        else:

            for AA in self.params:
                AA **= A

            return self

    # Negation
    def __neg__(self):

        params = [-p for p in self.params]

        return Params(params)

    # Slicing
    def __getitem__(self,sliced):

        return self.params[sliced]

    # Absolute value
    def __abs__(self):

        return Params([abs(A) for A in self.params])

    # Length
    def __len__(self):

        return len(self.params)

    # Sum of all parameters
    def sum(self):

        return sum([A.sum() for A in self.params])

    # Saves parameters to f as dtype if given
    def tofile(self,f,dtype=None):

        if isinstance(f,file):

            for p in self.params:
                if dtype is None:
                    p.tofile(f)
                else:
                    p.astype(dtype).tofile(f)

        else:

            with open(f,'w') as F:
                for p in self.params:
                    if dtype is None:
                        p.tofile(F)
                    else:
                        p.astype(dtype).tofile(F)

    # Return L2 norm of parameter vector
    def norm(self):

        return sqrt(sum([(p**2).sum() for p in self.params]))

    # Return copy with given dtype
    def astype(self,dtype):

        return Params([p.astype(dtype) for p in self.params])

