# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:22:03 2015

@author: rrowekamp
"""
from numpy import zeros,array,ones,exp,log,angle,mgrid,dot,pi,sqrt,inf,arange
from numpy.random import RandomState
from numpy.linalg import eigh,norm
from pickle import dump

# This class holds the sets of parameters and can generate new sets
class DEParams(object):

    # Either addtype or NP have to be given to give number of paramters
    def __init__(self,
                 func = None, # Cost function
                 addtype = None, # Type of addition for each parameter
                 pmin = None,
                 pmax = None,
                 NG = None, # Number of paramter sets (defaults to 10*NP)
                 FL = 0.1, # Determine range of scale parameter
                 FU = 0.9, # Randomly drawn from FL to FL + FU
                 NP = None, # Number of parameters per set
                 ):

        assert NP is not None or addtype is not None

        if NP is None:
             self.addtype = array(addtype).flatten()
             self.NP = self.addtype.size
        else:
             self.NP = NP
             if addtype is None:
                 self.addtype = zeros((self.NP,))
             else:
                 addtype = array(addtype).flatten()
                 if addtype.size == self.NP:
                     self.addtype = addtype
                 elif addtype.size == 1:
                     self.addtype = addtype*ones((self.NP,))
                 else:
                     raise Exception('Mismatch between NP and size of addtype')

        if pmin is not None:
            self.pmin = pmin.flatten()
        else:
            self.pmin = None
        if pmax is not None:
            self.pmax = pmax.flatten()
        else:
            self.pmax = None

        if NG is None:
            self.NG = self.NP*10
        else:
            self.NG = NG

        self.params = zeros((self.NG,self.NP))
        self.CR = zeros((self.NG,))
        self.F = zeros((self.NG,))
        self.FL = FL
        self.FU = FU
        assert isinstance(func,DEFunction) or func is None
        self.func = func
        self.error = zeros((self.NG,))

    # Generates random initial parameters
    def randomParams(self,
                     pmin=None, # Minimum value for each parameter
                     pmax=None,RS = None): # Maximum value for each parameter

         if pmin is not None:
             pmin = array(pmin).flatten()
             assert pmin.size == 1 or pmin.size == self.NP
         else:
             pmin = self.pmin

         if pmax is not None:
             pmax = array(pmax).flatten()
             assert pmax.size == 1 or pmax.size == self.NP
         else:
             pmax = self.pmax

         if not isinstance(RS,RandomState):
             RS = RandomState(RS)
         # Linear and angular variables are generated uniformly
         ind02 = (self.addtype == 0)+( self.addtype == 2)
         # Log variables are generated log uniformly
         ind1 = self.addtype == 1
         self.params[:,ind02] = (pmax[ind02]-pmin[ind02])*RS.rand(self.NG,ind02.sum()) + pmin[ind02]
         self.params[:,ind1] = exp((log(pmax[ind1])-log(pmin[ind1]))*RS.rand(self.NG,ind1.sum()) + log(pmin[ind1]))

         # Mutation rates uniformly distributed from 0 to 1
         self.CR = RS.rand(self.NG)
         # Scale parameters uniformly distributed from FL to FL+FU
         self.F = self.FL + self.FU*RS.rand(self.NG)

    # Generates new set of parameter from current parameters
    def mutate(self,
               TC = 0.1, # Probability of generating new CR for a parameter set
               TF = 0.1, # Probability of generating new F
               ND = 2): # Number of differences to use

        # Initialize object to hold new parameters
        child = DEParams(NP = self.NP,
                         addtype = self.addtype,
                         pmin = self.pmin,
                         pmax = self.pmax,
                         NG = self.NG,
                         FL = self.FL,
                         FU = self.FU,
                         func = self.func)

        # Copy CR and F
        child.CR = self.CR
        child.F = self.F

        # Get indices of different addtypes
        ind0 = self.addtype == 0 # Linear
        ind1 = self.addtype == 1 # Log
        ind2 = self.addtype == 2 # Angular

        # Generate new parameters
        RS = RandomState()
        for j in range(self.NG):
            # Select 2*ND+1 parameter sets to generate mutated values
            p = RS.permutation(self.NG-1)[:2*ND+1]
            # Make sure there is no inbreeding
            p[p>=j] += 1
            # Generate new CR and F if necessary
            if RS.rand() < TC:
                child.CR[j] = RS.rand()
            if RS.rand() < TF:
                child.F[j] = child.FL + child.FU*RS.rand()
            # Replace parameters with probability CR
            mask = RS.rand(self.NP)<child.CR[j]
            # Ensure at least one parameter is replaced
            mask[RS.randint(child.NP)] = True

            # Generate new parameters
            child.params[j,ind0] = (1-mask[ind0])*self.params[j,ind0] + mask[ind0]*(self.params[p[0],ind0]+child.F[j]*(self.params[p[1::2],:][:,ind0].sum(axis=0)-self.params[p[2::2],:][:,ind0].sum(axis=0)))
            child.params[j,ind1] = (1-mask[ind1])*self.params[j,ind1] + mask[ind1]*(exp(log(self.params[p[0],ind1])+child.F[j]*(log(self.params[p[1::2],:][:,ind1]).sum(axis=0)-log(self.params[p[2::2],:][:,ind1]).sum(axis=0))))
            child.params[j,ind2] = (1-mask[ind2])*self.params[j,ind2] + mask[ind2]*angle(exp(1j*(self.params[p[0],ind2]+child.F[j]*(self.params[p[1::2],:][:,ind2].sum(axis=0)-self.params[p[2::2],:][:,ind2].sum(axis=0)))))

        # Calculate costs for new parameter sets
        child.eval()

        return child

    # Calculates costs for parameter sets
    def eval(self):

        if self.func is not None:
            for j in range(self.NG):
                ok = True
                if self.pmin is not None:
                    ok = ok & all(self.params[j,:] > self.pmin)
                if self.pmax is not None:
                    ok = ok & all(self.params[j,:] < self.pmax)
                if ok:
                    try:
                        self.error[j] = self.func.eval(self.params[j,:])
                    except RuntimeWarning:
                        self.error[j] = inf
                else:
                    self.error[j] = inf

    # Take two DEParams objects and selects best set of parameters from each
    # corresponding pair. Creates new child if none is given.
    def merge(self,child = None):

        # Create new object
        new = DEParams(NP = self.NP,
                       addtype = self.addtype,
                       pmin = self.pmin,
                       pmax = self.pmax,
                       NG = self.NG,
                       FL = self.FL,
                       FU = self.FU,
                       func = self.func)

        # Create child if not given one
        if child is None:
            child = self.mutate()

        # Select groups where each is better
        ind1 = self.error > child.error
        ind2 = True - ind1

        # Put best sets into new object
        new.params[ind1,:] = child.params[ind1,:]
        new.params[ind2,:] = self.params[ind2,:]
        new.error[ind1] = child.error[ind1]
        new.error[ind2] = self.error[ind2]

        return new

    def paramsMin(self):

        return self.params[self.error.argmin(),:]

"""
Class to train DEParams
Inputs:
    func: DEFunction to fit
    DEP: DEParams to start from
    args: Arguments for parameter initialization
    NG: Number of parameter groups to use
    Bound: Force parameters to stay within initialization range
"""
class DETrainer(object):

    def __init__(self,
                 func,
                 DEP = None,
                 args = (),
                 NG = None,
                 Bound = False):

        assert isinstance(func,DEFunction)
        self.func = func

        self.Bound = Bound
        self.NG = NG

        if DEP is None:
            self.addtype,self.pmin,self.pmax = self.func.paramsInit(*args)
            if self.Bound:
                self.DEP = DEParams(self.func,self.addtype,self.pmin,
                                    self.pmax,self.NG)
            else:
                self.DEP = DEParams(self.func,self.addtype,NG=self.NG)
        else:
            assert isinstance(DEP,DEParams)
            self.DEP = DEP

    """
    Optimizes function using differential evolution steps.
    Inputs:
        NT: Max number of steps
        reset: Reset parameters
        verbose: Output updates while training if True
        mindelta: Minimum improvement in mean error to continue training
        saveFile: File to save current state after every iteration
    """
    def train(self,NT=inf,reset=True,verbose=False,mindelta=0,saveFile=None):

        if reset:
            if self.Bound:
                self.DEP.randomParams()
            else:
                self.DEP.randomParams(self.pmin,self.pmax)
            self.DEP.eval()
        e0 = self.DEP.error.mean()
        self.DEP = self.DEP.merge()
        e1 = self.DEP.error.mean()
        its = 0
        while its < NT and e1 - e0 < -mindelta*e0:
            its += 1
            e0 = e1
            self.DEP = self.DEP.merge()
            e1 = self.DEP.error.mean()
            if verbose:
                print its,e1
            if saveFile is not None:
                with open(saveFile,'w') as f:
                    dump(self,f)


# Cost function used by DEParams
class DEFunction(object):

    def __init__(self):
        raise NotImplementedError()

    # Evalutes cost for each parameter set
    def eval(self,params):
        raise NotImplementedError()

    # Gives pmin, pmax, and addtype
    def paramsInit(self):
        raise NotImplementedError()

# Fits symmetric matrix using quadrature pairs of Gabors
class JGaborPairFunction(DEFunction):

    def __init__(self,
                 J):

         self.J = J
         self.J.shape = 2*(sqrt(self.J.size),)
         self.NX = sqrt(sqrt(self.J.size))
         x,y = mgrid[:self.NX,:self.NX]
         self.z = x+1j*y
         self.z.shape = (-1,)
         self.J0 = (self.J**2).mean()

    def eval(self,params):

         JG = self.makeJ(params)

         return ((self.J-JG)**2).mean()/self.J0

    def paramsInit(self,ng):

         addtype = zeros((7,ng))
         pmin = zeros((7,ng))
         pmax = zeros((7,ng))

         wmax = abs(eigh(self.J)[0]).max()

         pmin[0,:] = -2*wmax
         pmax[0,:] = 2*wmax
         addtype[0,:] = 0

         pmin[1:3,:] = 0.5
         pmax[1:3,:] = self.NX - 1.5
         addtype[1:3,:] = 0

         pmin[3,:] = -pi
         pmax[3,:] = pi
         addtype[3,:] = 2

         pmin[4,:] = 2
         pmax[4,:] = self.NX/3.
         addtype[4,:] = 1

         pmin[5,:] = 2./3.
         pmax[5,:] = 3./2.
         addtype[5,:] = 1

         pmin[6,:] = 4.
         pmax[6,:] = self.NX/1.5
         addtype[6,:] = 1

         return addtype,pmin,pmax

    def makeG(self,params):

         p = params.copy()
         p.shape = (7,-1)

         g = self.z.copy()
         g.shape = (-1,1)

         g = g - (p[1,:]+p[2,:]*1j)
         g = g * exp(1j*p[3,:])

         g = exp(-(g.real**2+g.imag**2*p[5,:]**2)/2/p[4,:]**2)*exp(1j*2*pi/p[6,:]*g.real)
         g = g / sqrt(abs(g*g.conj()).sum(axis=0))

         return g,p[0,:]

    def makeJ(self,params):

         g,p = self.makeG(params)

         JG = dot(g.real*p,g.real.T)
         JG += dot(g.imag*p,g.imag.T)

         return JG

# Fits symmetric matrix using Gabors
class JGaborFunction(DEFunction):

    def __init__(self,
                 J):

         self.J = J
         self.J.shape = 2*(sqrt(self.J.size),)
         self.NX = sqrt(sqrt(self.J.size))
         x,y = mgrid[:self.NX,:self.NX]
         self.z = x+1j*y
         self.z.shape = (-1,)
         self.J0 = (self.J**2).mean()

    def makeG(self,params):

        p = params.copy()
        p.shape = (8,-1)

        g = self.z.copy()
        g.shape = (-1,1)

        g = g - (p[1,:]+p[2,:]*1j)
        g = g * exp(1j*p[3,:])

        g = exp(-(g.real**2+g.imag**2*p[5,:]**2)/2/p[4,:]**2)*exp(1j*2*pi/p[6,:]*g.real)
        g *= exp(1j*p[7,:])
        g = g / sqrt(abs(g*g.conj()).sum(axis=0))

        return g,p[0,:]

    def makeJ(self,params):

        g,p = self.makeG(params)
        return dot(g.real*p,g.real.T)

    def eval(self,params):

         JG = self.makeJ(params)

         return ((self.J-JG)**2).mean()/self.J0

    def paramsInit(self,ng):

         addtype = zeros((8,ng))
         pmin = zeros((8,ng))
         pmax = zeros((8,ng))

         wmax = abs(eigh(self.J)[0]).max()

         pmin[0,:] = -2*wmax
         pmax[0,:] = 2*wmax
         addtype[0,:] = 0

         pmin[1:3,:] = 1.5
         pmax[1:3,:] = self.NX - 2.5
         addtype[1:3,:] = 0

         pmin[3,:] = -pi
         pmax[3,:] = pi
         addtype[3,:] = 2

         pmin[4,:] = 2
         pmax[4,:] = self.NX/4
         addtype[4,:] = 1

         pmin[5,:] = 2./3.
         pmax[5,:] = 3./2.
         addtype[5,:] = 1

         pmin[6,:] = 4.
         pmax[6,:] = self.NX/2.
         addtype[6,:] = 1

         pmin[7,:] = -pi
         pmax[7,:] = pi
         addtype[7,:] = 2

         return addtype,pmin,pmax

# Fit matrix as a Gabor
class Gabor2DFunction(DEFunction):

    def __init__(self,X):

        self.NX = X.shape
        self.X = X.flatten()
        x,y = mgrid[:self.NX[0],:self.NX[1]]
        self.Z = x.flatten()+1j*y.flatten()

    def paramsInit(self):

        addtype = zeros(7)
        pmin = zeros(7)
        pmax = zeros(7)

        pmin[:2] = -0.5
        pmax[0] = self.NX[0]-0.5
        pmax[1] = self.NX[1]-0.5
        addtype[:2] = 0

        pmin[2] = 0.5
        pmax[2] = sqrt(self.NX[0]**2+self.NX[1]**2)
        addtype[2] = 1

        pmin[3] = -pi
        pmax[3] = pi
        addtype[3] = 2

        pmin[4] = 0.5
        pmax[4] = 2
        addtype[4] = 1

        pmin[5] = 3.
        pmax[5] = sqrt(self.NX[0]**2+self.NX[1]**2)
        addtype[5] = 1

        pmin[6] = -pi
        pmax[6] = pi
        addtype[6] = 2

        return addtype,pmin,pmax

    def makeG(self,params):

        G = self.Z.copy()
        G -= params[0] + params[1]*1j
        G *= exp(1j*params[3])

        G = exp(-(G.real**2+G.imag**2*params[4]**2)/2/params[2]**2)*exp(1j*(G.real*2*pi/params[5]+params[6])).real

        return G

    def eval(self,params):

        G = self.makeG(params)

        return 1-abs(dot(G,self.X)/norm(G)/norm(self.X))

# Fit vector using Gaussians
class Gauss1DFunction(DEFunction):

    def __init__(self,X):

        self.X = X
        self.NX = X.size
        self.x0 = arange(self.NX)

    def paramsInit(self,ng):

        addtype = zeros((3,ng))
        pmin = zeros((3,ng))
        pmax = zeros((3,ng))

        pmin[0,:] = -abs(self.X).max()
        pmax[0,:] = -pmin[0,:]
        addtype[0,:] = 0

        pmin[1,:] = -0.5
        pmax[1,:] = -0.5 + self.NX
        addtype[1,:] = 0

        pmin[2,:] = 0.1
        pmax[2,:] = 2*self.NX
        addtype[2,:] = 1

        return addtype,pmin,pmax

    def makeG(self,params):

        p = params.reshape((3,-1))
        g = self.x0[:,None] - p[1,:]
        g **= 2
        g /= p[2,:]**2
        g = exp(-g)*p[0,:]

        return g

    def eval(self,params):

        g = self.makeG(params)

        return ((self.X-g.sum(1))**2).sum()

# Fit a matrix as a Gaussian
class Gauss2DFunction(DEFunction):

    def __init__(self,X):

        self.NX = X.shape
        self.X = X.flatten()
        x,y = mgrid[:self.NX[0],:self.NX[1]]
        self.Z = x.flatten()+1j*y.flatten()

    def paramsInit(self):

        addtype = zeros(5)
        pmin = zeros(5)
        pmax = zeros(5)

        pmin[:2] = -0.5
        pmax[0] = self.NX[0]-0.5
        pmax[1] = self.NX[1]-0.5
        addtype[:2] = 0

        pmin[2] = 0.5
        pmax[2] = sqrt(self.NX[0]**2+self.NX[1]**2)
        addtype[2] = 1

        pmin[3] = -pi
        pmax[3] = pi
        addtype[3] = 2

        pmin[4] = 0.5
        pmax[4] = 2
        addtype[4] = 1

        return addtype,pmin,pmax

    def makeG(self,params):

        G = self.Z.copy()
        G -= params[0] + params[1]*1j
        G *= exp(1j*params[3])

        G = exp(-(G.real**2+G.imag**2*params[4]**2)/2/params[2]**2)

        return G

    def eval(self,params):

        G = self.makeG(params)

        return 1-abs(dot(G,self.X)/norm(G)/norm(self.X))
