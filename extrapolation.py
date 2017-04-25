# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:21:38 2015

@author: rrowekamp
"""

from numpy import corrcoef,arange,zeros,ones,dot,unique,sqrt,median
from numpy.linalg import inv
from numpy.random import permutation

# 1/correlation**2 extrapolates linearly with 1/N
def icorr2(a,b):
    return 1./corrcoef(a,b)[1,0]**2

"""
Calcualtes mean response for each stimulus given a list of responses and
stimulus IDs.
Inputs:
    Y: The responses for each observation.
    ID: The stimulus IDs for each observation.
Returns averaged responses
"""
def YbarID(Y,ID):

    uID = unique(ID)

    YB = zeros(Y.shape)

    for j in range(uID.size):
        ind = ID==uID[j]
        YB[ind] = Y[ind].mean()

    return YB

"""
Extrap class takes predictions, observations, stimulus IDs, and a function and
extrapolates the function to infinite data.
Inputs:
    r: Predicted responses
    Y: Observed responses
    ID: IDs of the associated stimuli
    func: The function to be extrapolated
"""
class Extrap(object):

    def __init__(self,r,Y,ID,func):

        self.r = r

        assert Y.shape == r.shape
        self.Y = Y

        assert ID.shape == Y.shape
        self.ID = ID

        self.func = func
        self.uID = unique(ID)

    # Creates an Extrap object with a subsample of the data
    def SubSample(self,ind):

        return Extrap(self.r[ind],self.Y[ind],self.ID[ind],self.func)

    # Return boolean array with N random indices set to False
    def TrialFrac(self,N):

        p = permutation(self.Y.size)

        try:
            p = p[:N]
        except TypeError:
            print 'Type Error in Extrap.TrialFrac.'
            print type(N)
            print N
            p = p[:int(N)]

        a = ones(self.Y.size,dtype=bool)
        a[p] = False
        return a

    """
    Extrapolates function to infinite data
    Inputs:
        frac: The fractions of the data to omit
        nrep: The number of repitions to do for each fraction
        doMedian: Use median value for each fraction
        fullOutput: Output estimate of variance and values used for extrapolation
    Returns:
        Extrapolated value
        If fullOutput is True, also returns variance of estimated extrapolation
        and the values used in the extrapolations
    """
    def extrap(self,frac=0.05*arange(5),nrep=100,doMedian=True,fullOutput=False):

        # Convert fractions into numbers of samples
        frac = (self.Y.size*frac.flatten()).astype(int)

        # Convert frac into array with all of the repetitions
        # The case where frac=0 is unique, so it is only calculated once
        nf = frac.size
        n0 = (frac==0).sum()
        frac1 = frac
        frac = zeros(n0+(nf-n0)*nrep,'int')
        for j in range(n0,nf):
            frac[n0+(j-n0)*nrep:n0+(j-n0+1)*nrep] = frac1[j]

        # Calculate function value given random ommissions of samples
        C = zeros(frac.size)
        for j in range(frac.size):
            p = self.TrialFrac(frac[j])
            YB = YbarID(self.Y[p],self.ID[p])

            C[j] = self.func(self.r[p],YB)

        # If doMedian, caculate median for each fraction
        if doMedian:
            c = C
            C = zeros(frac1.size)
            for j in range(frac1.size):
                C[j] = median(c[frac==frac1[j]])
            frac = frac1

        # Do linear regression
        X = ones((frac.size,2))
        X[:,1] = 1./(self.Y.size-frac)

        B = dot(inv(dot(X.T,X)),dot(X.T,C))

        # If fullOutput, calculate variance in estimated intercept
        if fullOutput:
            SB = B.copy()
            SB[1] = sqrt(((dot(X,B)-C)**2).sum()/(C.size-2)/((X[:,1]-X[:,1].mean())**2).sum())
            SB[0] = SB[1]*sqrt((X[:,1]**2).sum()/C.size)
            SB = SB[0]
        B = B[0]

        if fullOutput:
            return B,SB,(X,C)

        else:
            return B

