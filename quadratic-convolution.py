from numpy import tensordot as tdot
from numpy import zeros,ones,array,ndarray,delete,dot,arange
from numpy import spacing,prod,fromfile,exp,log,inf,isfinite
from numpy.linalg import norm,inv
from sys import stdout
from os import remove
from os.path import exists,expanduser,isdir
from time import time
from numpy.random import RandomState
from numpy.lib.stride_tricks import as_strided
from warnings import filterwarnings
import warnings
from Params import Params
from types import IntType,LongType,FloatType,StringType
NumType = (IntType,LongType,FloatType)

# Small number
eps = spacing(1.)

"""
Normalize stimulus
    stim: numpy array with sample number as last dimension
    pixelNorm: If True, normalize each location by individual mean and stdev.
       If a tuple, use first element as mean and second as stdev.
       Otherwise, normalize using global statistics.
    Returns normalized stimulus, mean, and stdev.
"""
def normStim(stim,pixelNorm=True):

    # Ignore divide by zero warnings
    filterwarnings('ignore','invalid value encountered in divide')

    # Normalize using given values
    if isinstance(pixelNorm,tuple):
        # Should contain two values: mean, stdev
        assert len(pixelNorm) == 2
        stimAve,stimStDev = pixelNorm

        # Convert number to numpy array
        if isinstance(stimAve,int) or isinstance(stimAve,long) or isinstance(stimAve,float):
            stimAve = array(stimAve)
        # Otherwise, make sure input is in a compatible shape
        elif isinstance(stimAve,ndarray):
            assert stimAve.shape == stim.shape[:-1]+(1,) or stimAve.size == 1

    # Normalize using pixel statistics
    elif pixelNorm:
        stimAve = stim.mean(axis=-1)
        stimAve.shape += (1,)
        stimStDev = stim.std(axis=-1)
        stimStDev.shape += (1,)

    # Normalize using full stimulus statistics
    else:
        stimAve = stim.mean()
        stimStDev = stim.std()

    # Normalize stim
    stim -= stimAve
    stim /= stimStDev

    # Check for bad pixels (usually pixel with no variation)
    stim[~isfinite(stim)]=0.

    # Remove filter added above so it does not affect later code
    warnings.filters = warnings.filters[1:]

    return stim,stimAve,stimStDev

# Check if x is an integer
def IntCheck(x):
    if isinstance(x,IntType):
        return x
    else:
        assert x == int(x)
        return int(x)

# Soft-rectifier (log(1+exp(x)))
def softPlus(x):

    # Create copy of x
    r = array(x)

    # If exp(r) would overflow, treat softPlus(r) as linear
    if r.ndim:
        r[r<700] = log(1+exp(r[r<700]))
    else:
        if r < 700:
            r = log(1+exp(r))

    return r

# Derivative of softplus rectifier
def dSP(x):

    return logistic(x)

# Array version of logistic function
def logistic(x):

    return 1/(1+exp(-x))

# Derivative of logistic function
def dlog(x):

    x = logistic(x)

    return x*(1-x)

# Create collection of patches
def gridStim(stim,fsize,nlags=1):

    FSIZE = stim.shape[:-1]+(nlags,)
    gsize = tuple([F-f+1 for F,f in zip(FSIZE,fsize)])

    ssh = stim.shape
    sst = stim.strides

    Ssh = (ssh[-1]-nlags+1,)+fsize+gsize
    Sst = sst[-1:]+2*sst

    return as_strided(stim,shape=Ssh,strides=Sst)

# Calculate gradient for softplus model
def gradSP(Y, # Observed response
           S, # Stimulus
           P  # Parameters
           ):

    # Extract parameters
    a1,v1,J1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    # Derivatives of model nonlinearities and cost function
    df1,df2,dfe = dlog,dSP,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(range(ndim),))+(tdot(J1,S,2*(range(ndim),))*S).sum(tuple(range(ndim)))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(range(-ndim,0),range(ndim)))
    dJ1 = dy*dr2*tdot(dr1*S*S.reshape(S.shape[:ndim]+ndim*(1,)+S.shape[ndim:]),v2,(range(-ndim,0),range(ndim)))

    return Params([da1,dv1,dJ1,da2,dv2,dd])

# Calculate gradient for linear softplus model
def gradLinearSP(Y, # Observed response
                 S, # Stimulus
                 P  # Parameters
                 ):

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    # Derivative of model nonlinearities and cost function
    df1,df2,dfe = dlog,dSP,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(range(ndim),))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(range(-ndim,0),range(ndim)))

    return Params([da1,dv1,da2,dv2,dd])

# Calculate gradient for logistic model
def gradLog2(Y, # Observed response
             S, # Stimulus
             P  # Parameters
             ):

    # Extract parameters
    a1,v1,J1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    # Derivative of model nonlinearities and cost function
    df1,df2,dfe = dlog,dlog,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(range(ndim),))+(tdot(J1,S,2*(range(ndim),))*S).sum(tuple(range(ndim)))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(range(-ndim,0),range(ndim)))
    dJ1 = dy*dr2*tdot(dr1*S*S.reshape(S.shape[:ndim]+ndim*(1,)+S.shape[ndim:]),v2,(range(-ndim,0),range(ndim)))

    return Params([da1,dv1,dJ1,da2,dv2,dd])

# Calculate gradient for linear logistic model
def gradLinearLog2(Y, # Observed response
                   S, # Stimulus
                   P  # Parameters
                   ):

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    # Derivative of model nonlinearities and cost function
    df1,df2,dfe = dlog,dSP,dllike

    ndim = v1.ndim

    x1 = a1+tdot(v1,S,2*(range(ndim),))
    r1 = f1(x1)
    dr1 = df1(x1)
    x2 = a2+(r1*v2).sum()
    r2 = f2(x2)
    dr2 = df2(x2)

    dy = d*dfe(Y,d*r2)
    dd = dy*r2/d

    da2 = dy*dr2
    dv2 = dy*dr2*r1

    da1 = dy*dr2*(dr1*v2).sum()
    dv1 = dy*dr2*tdot(dr1*S,v2,(range(-ndim,0),range(ndim)))

    return Params([da1,dv1,da2,dv2,dd])

# Calculates responses for softplus model
def respSP(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Model parameters
    # Output:
    #   r - Response of model for given stimulus

    # Extract parameters
    a1,v1,J,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(range(ndim),))+(tdot(J,S,2*(range(ndim),))*S).sum(tuple(range(ndim))))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculates response for linear softplus model
def respLinearSP(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Parameters
    # Output:
    #   r - Response of model for give stimulus

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,softPlus

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(range(ndim),)))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculates responses for logistic model
def respLog2(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Model parameters
    # Output:
    #   r - Response of model for given stimulus

    # Extract parameters
    a1,v1,J,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(range(ndim),))+(tdot(J,S,2*(range(ndim),))*S).sum(tuple(range(ndim))))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculates responses for linear logistic model
def respLinearLog2(S,P):
    # Inputs:
    #   S - A stimulus
    #   P - Model parameters
    # Output:
    #   r - Response of model for given stimulus

    # Extract parameters
    a1,v1,a2,v2,d = P

    # Model nonlinearities
    f1,f2 = logistic,logistic

    ndim = v1.ndim

    # Calculate first layer responses
    r1 = f1(a1+tdot(v1,S,2*(range(ndim),)))

    # Calculate second layer responses
    r2 = f2(a2+(r1*v2).sum())

    return d*r2

# Calculate responses of many stimuli
def Resp(S,P,func):
    r = array([func(s,P) for s in S])
    rs = r.shape[:1]+r.shape[2:]
    return r.reshape(rs)

# Poisson log likelihood
# Returns difference between likelihood of predictions and observations in order
# to make value positive
def llike(Y,R):
    return (Y*log(Y+eps)-Y-Y*log(R+eps)+R).mean()

# Derivative of poisson log-likelihood
def dllike(Y,R):
    return Y/(R+eps)-1

"""
LearningRate class to adjust learning rate
Base class keeps learning rate constant unless training error increases.
"""
class LearningRate(object):

    def __init__(self,
                 initialError, # Initial training error
                 initialRate = 1e-4, # Initial learning rate
                 lrDown = 0.1, # Learning rate multiplied by this factor if error increases
                 **kwargs):

        self.lrate = initialRate
        self.lastError = initialError
        self.lrDown = lrDown

    # Checks training error and decreases learning rate if error has increased
    def update(self,error,*args,**kwargs):

        if error < self.lastError:
            self.lastError = error
        else:
            self.lrate *= self.lrDown

# Learning rate decays as l0/(1+its/tau)
# Decreases l0 if training error increases
class DecayRate(LearningRate):

    def __init__(self,
                 initialError, # Initial training error
                 initialIts = 0, # Initial value of its
                 initialRate = 1e-4, # Initial learning rate
                 lrTau = 1, # Time constant of decay
                 lrDown = 0.1, # Multiplies initialRate if erorr increased
                 **kwargs):

         self.initialRate = initialRate
         self.its = initialIts
         self.lrTau = lrTau
         self.lrate = self.initialRate/(1+self.its/self.lrTau)
         self.lrDown = 0.1

         self.lastError = initialError

    # Reduces initialRate if error has increased. Updates learning rate.
    def update(self,error,*args,**kwargs):

         if error < self.lastError:
             self.its += 1
             self.lastError = error
         else:
             self.initialRate *= self.lrDown

         self.lrate = self.initialRate/(1+self.its/self.lrTau)

# Increases learning rate if error decreasing. Decreases learning rate if erorr increasing
class BoldDriver(LearningRate):

    def __init__(self,
                 initialError, # Initial error
                 initialRate = 1e-2, # Initial learning rate
                 learningUp = 5e-2, # Amount to increase rate if error decreasing
                 learningDown = 0.5, # Amount to decrease rate if error increasing
                 **kwargs):

        self.lrate = initialRate
        self.lUp = learningUp
        self.lDown = learningDown

        self.lastError = initialError

    # Increases or decreases rate if error is decreasing or increasing (respectively)
    def update(self,error,*args,**kwargs):

        if error < self.lastError:
            self.lrate += self.lUp*self.lrate
            self.lastError = error
        else:
            self.lrate *= self.lDown

"""
This function either initializes the model or loads previous run and fits it
with the given stimuli and responses. Outputs file with best parameters on
training set, file with best parameters on validation set, file with training
error history, and file with validation error history. Also creates status
file used to restart incomplete runs that is deleted when the fit converges.
Inputs:
    prefix: String appended to all output files
    spikes: numpy array of responses to predict
    stim: numpy array of stimuli. The last dimension is the sample number.
    jack: Jackknife used for validation. From 1 to Njack
    fsize: tuple with shape of the first layer's filter size
    extrapSteps: Number of steps used to estimate slope of validation error
    pixelNorm: Whether to normalize using local statistics (if True) or global
        (if False)
    filepath: Path where to save output files
    model: Type of model to fit
    maxIts: Maximum number of iterations to run
    maxHours: Maximum number of hours to run
    perm: Whether to randomly permute stimulus-response pairs before divinding
        into training and validation sets
    overwrite: Whether to overwrite an existing completed fit
    Njack: The validation set is 1/Njack of the data
    start: How to initialize the parameters. Can be Params object, list/tuple
        of numpy arrays, or a string. If a string, options for first and
        second layers are seperated by '_'. Options for first layer are rand
        (random initialization), sta (STA intialization), and stim (intialization
        from random combination of stimuli). Options for second layer are rand
        (random initialization), sta (STA initialization), and uniform (uniform
        initialization).
    nlags: Number of time frames from the stimulus used to predict response.
    splits: Locations of splits in the stimuli/responses. Used for nlags > 1 so
        that stimuli from one recording aren't used to predict responses for
        the next recording.
    LRType: Learning rate rule used.
    LRParams: Parameters for learning rate rule.
"""
def gradDescent(prefix,spikes,stim,jack,fsize,extrapSteps=10,
                pixelNorm=True,filepath=None,model='softplus',
                maxIts=None,maxHours=None,perm=True,overwrite=False,
                Njack=4,start='rand_rand',nlags=1,splits=None,
                LRType='DecayRate',LRParams = {}):


    assert isinstance(prefix,StringType)
    print 'Prefix ' + prefix

    Njack = IntCheck(Njack)
    jack = IntCheck(jack)
    assert jack > 0 and jack <= Njack
    print 'Jack ',jack,'out of ',Njack

    FSIZE = stim.shape[:-1]+(nlags,)
    print 'Full frame size ',FSIZE
    assert isinstance(fsize,tuple)
    if len(fsize) < len(FSIZE):
        fsize = fsize + (len(FSIZE)-len(fsize))*(1,)
    print 'Patch frame size ',fsize
    gsize = tuple([F-f+1 for F,f in zip(FSIZE,fsize)])
    NGRID = prod(gsize)
    ng = len(gsize)
    print 'Grid size ',gsize

    assert model in ['softplus','linearSoftplu','logistic','linearLogistic']
    print 'Model ',model
    if model == 'softplus':
        resp = respSP
        grad = gradSP
        cost = llike
        AlgTag = '_QuadraticSoftPlus'
    elif model == 'linearSoftplus':
        resp = respLinearSP
        grad = gradLinearSP
        cost = llike
        AlgTag = '_LinearSoftPlus'
    elif model == 'logistic':
        resp = respLog2
        grad = gradLog2
        cost = llike
        AlgTag = '_QuadraticLogistic'
    elif model == 'linearLogistic':
        resp = respLinearLog2
        grad = gradLinearLog2
        cost = llike
        AlgTag = '_LinearLogistic'

    extrapSteps = IntCheck(extrapSteps)
    print 'Steps used to estimate error slipe ',extrapSteps

    if pixelNorm:
        print 'Normalizing by pixel statistics'
    else:
        print 'Normalizing by global statistics'

    if filepath in [None,'','./']:
        filepath=''
        print 'Saving output in current directory'
    else:
        assert isinstance(filepath,StringType)
        filepath = expanduser(filepath)
        assert isdir(filepath)
        print 'Saving output files to ',filepath

    assert isinstance(start,list) or isinstance(start,tuple) or isinstance(start,str) or isinstance(start,Params)
    if isinstance(start,str):
        vstart,bstart = start.split('_')
        assert bstart in ['rand','sta','uniform']
        assert vstart in ['rand','stim','sta']
        if bstart == 'rand':
            print 'Initializing second layer randomly'
        elif bstart == 'unifrom':
            print 'Initializing second layer uniformly'
        elif bstart == 'sta':
            print 'Initializing second layer using STA'
        if vstart == 'rand':
            print 'Initializing first layer randomly'
        elif vstart == 'stim':
            print 'Initializing first layer using random stimuli'
        elif vstart == 'sta':
            print 'Initializing first layer using STA'
    else:
        print 'Starting parameters given'

    if maxIts is None:
        maxIts = inf
        print 'No limit on iterations'
    else:
        maxIts = IntCheck(maxIts)
        print 'Max iterations ',maxIts

    genesis = time()
    if maxHours is None:
        print 'No limit on runtime'
        eschaton = inf
    else:
        assert isinstance(maxHours,NumType)
        eschaton = genesis + maxHours*3600
        print 'Max hours ',maxHours

    if isinstance(perm,ndarray):
        print 'Permuting data by given array before division'
    else:
        if perm:
            print 'Randomly divided data sets'
        else:
            print 'Contiguous data sets'

    # Get stimulus shape and size
    Ntrials = stim.shape[-1]-nlags+1

    # Convert spikes from int
    Y = spikes.astype(float)
    del spikes

    # Drop spikes before first full stimulus
    Y = Y[nlags-1:]

    # Check that stimulus and responses have the same size
    assert Y.size == Ntrials

    # Size of first layer input
    npix = prod(fsize)

    # Convert stimulus to zero mean and unit stdev
    stim = normStim(stim,pixelNorm)[0]

    Nvalid = Ntrials / Njack
    Ntrials -= Nvalid

    # Randomly permute stimulus and spikes
    if isinstance(perm,ndarray):
        assert perm.size == Ntrials+Nvalid
        p = perm[nlags-1:]
    elif perm:
        RS = RandomState(0)
        p = RS.permutation(Ntrials+Nvalid-nlags+1)

    # Split data into training and test sets
    validslice = slice((jack-1)*Nvalid,jack*Nvalid)
    pv = p[validslice]
    pr = delete(p,validslice)

    # Remove samples that span recordings from training and validation sets
    if splits is not None:
        invalid = array([arange(sp-nlags+1,sp) for sp in splits]).flatten()
        pv = array([pp for pp in pv if pp not in invalid])
        pr = array([pp for pp in pr if pp not in invalid])

    # Extract stimulus at grid locations
    S = gridStim(stim,fsize,nlags)

    # Divide responses into training and validation sets
    YR = Y[pr]
    YV = Y[pv]

    spikesmean = YR.mean()
    Nspikes = YR.sum()

    # Calcualte error of mean model
    errTrain0 = cost(YR,spikesmean)
    errValid0 = cost(YV,spikesmean)

    stdout.write('Training: {0} frames, {1} spikes\n'.format(Ntrials,Nspikes))
    stdout.write('Validation: {0} frames, {1} spikes\n'.format(Nvalid,YV.sum()))
    stdout.flush()

    # Create filenames
    trainBestName = filepath+prefix+AlgTag+'_train_%u.dat' % (jack,)
    validBestName = filepath+prefix+AlgTag+'_valid_%u.dat' % (jack,)
    statusName = filepath+prefix+AlgTag+'_%u.temp' % (jack,)
    errTrainName = filepath+prefix+AlgTag+'_errTrain_%u.dat' % (jack,)
    errValidName = filepath+prefix+AlgTag+'_errValid_%u.dat' % (jack,)

    # Calculate shapes of parameters
    if model in ['softplus','logistic']:
        shapes = [(1,),fsize,2*fsize,(1,),gsize,(1,)]
    else:
        shapes = [(1,),fsize,(1,),gsize,(1,)]

    # Check to see if previous run exists
    if exists(statusName):

        stdout.write('Loading previous run\n')
        stdout.flush()
        with open(statusName,'r') as f:
            its = fromfile(f,count=1,dtype=int)
            errValidMin = fromfile(f,count=1)
        P = Params(trainBestName,shapes)
        PV = Params(validBestName,shapes)
        if its > maxIts:
            maxIts += its
        with open(errValidName,'r') as f:
            errValidHist = list(fromfile(f))
        if len(errValidHist) > extrapSteps:
            errValidHist = errValidHist[-extrapSteps:]
        with open(errTrainName,'r') as f:
            errTrain = fromfile(f)[-1]
    else:
        if exists(trainBestName) and not overwrite:
            print 'Output files exist'
            return
        else:
            # If start is a Params object, copy it
            if isinstance(start,Params):
                P = start.copy()
            # If start is a list/tuple, reshape values and convert to Params
            elif isinstance(start,list) or isinstance(start,tuple):
                assert len(start) == len(shapes)
                for s,p in zip(shapes,start):
                    p.shape = s
                P = Params(start,shapes)
            else:
                # Initialize first layer randomly
                if vstart == 'rand':
                    RS = RandomState()
                    v = RS.randn(npix).reshape(fsize)
                    v /= norm(v)
                    if model in ['softplus','logistic']:
                        J = RS.randn(npix,npix)
                        J = J+J.T
                        J /= norm(J)
                        J.shape = 2*fsize

                # Initialize first layer with random stimuli from training set
                elif vstart == 'stim':
                    RS = RandomState()
                    v = zeros(fsize)
                    for j in pr:
                        r = RS.randn(NGRID).reshape(gsize)
                        v += tdot(S[j,...],r,(range(-ng,0),range(ng)))
                    v /= norm(v)
                    if model in ['softplus','logistic']:
                        J = zeros(2*fsize)
                        for j in pr:
                            r = RS.randn(NGRID).reshape(gsize)
                            J += tdot(S[j,...],S[j,...]*r,(range(-ng,0),range(-ng,0)))
                        J /= norm(J)

                # Initialize first layer using STA/STC
                elif vstart == 'sta':
                    ES = zeros(fsize)
                    ESY = zeros(fsize)
                    if model in ['softplus','logistic']:
                        ESS = zeros(fsize*2)
                        ESSY = zeros(fsize*2)
                    for pp in pr:
                        SS = S[pp,...].sum(-1).sum(-1).sum(-1).sum(-1)
                        ES += SS
                        ESY += SS*Y[pp]
                        if model in ['softplus','logistic']:
                            SSS = SS*SS.reshape(SS.shape+4*(1,))
                            ESS += SSS
                            ESSY += SSS*Y[pp]
                    ES /= pr.size
                    ESY /= YR.sum()
                    v = ESY - ES
                    v /= norm(v)
                    if model in ['softplus','logistic']:
                        ESS /= pr.size
                        ESSY /= YR.sum()
                        J = (ESSY-ESY*ESY.reshape(ESY.shape+4*(1,)))-(ESS-ES*ES.reshape(ES.shape+4*(1,)))
                        J /= norm(J)

                else:
                    raise Exception('Unsupported initialization')

                # Scale v and J.
                v *= 0.1
                if model in ['softplus','logistic']:
                    J *= 0.1

                # Initialize second layer randomly
                if bstart == 'rand':
                    RS = RandomState()
                    v2 = RS.randn(*gsize)
                    v2 /= norm(v2)
                    v2 *= 0.1

                # Initialize second layer uniformly
                elif bstart == 'uniform':
                    v2 = ones(gsize)
                    v2 /= norm(v2)
                    v2 *= 0.1

                # Intialize second layer using STA
                elif bstart == 'sta':
                    ES = zeros(gsize)
                    ESY = zeros(gsize)
                    for pp in pr:
                        xv = tdot(S[pp,...],v,2*(range(4),))
                        xJ = (tdot(J,S[pp,...],2*(range(4),))*S[pp,...]).sum(0).sum(0).sum(0).sum(0)
                        r1 = logistic(xv+xJ)
                        ES += r1
                        ESY += r1*Y[pp]
                    ES /= pr.size
                    ESY /= YR.sum()
                    v2 = ESY-ES
                    v2 /= norm(v2)
                    v2 *= 0.1

                # Combine intialized parameters into a Params object
                if model in ['softplus','logistic']:
                    P = Params([zeros(1),v,J,zeros(1),v2,ones(1)])
                else:
                    P = Params([zeros(1),v,zeros(1),v2,ones(1)])

                # Set d to match mean firing rate on training set
                R = array([resp(S[j,...],P) for j in pr])
                rmean = R.mean()
                P[-1][:] = spikesmean/rmean
                P = Params(P)

            # Calculate initial error
            R = Resp(S,P,resp)
            errTrain = cost(YR,R[pr])/errTrain0
            errValid = cost(YV,R[pv])/errValid0

            # Save initial errors
            with open(errTrainName,'w') as f:
                errTrain.tofile(f)

            with open(errValidName,'w') as f:
                errValid.tofile(f)

            errValidHist = [errValid]

            # Save initial values as best so far
            errValidMin = errValid.copy()
            PV = P.copy()

            # Save initial parameters to parameter files
            P.tofile(trainBestName)
            PV.tofile(validBestName)

            # Keep track of the number of iterations
            its = 0

    stdout.write('Beginning optimization\n')
    stdout.flush()
    if model in ['softplus','logistic']:
        Pname = ['a1','v1','J1','a2','v2','d']
    else:
        Pname = ['a1','v1','a2','v2','d']

    # Start slope as negative
    slope = -1.

    # Print status
    print '%u Values:' % (its,),
    for nam,p in zip(Pname,P):
        if p.size == 1:
            print ' %s %.3e' % (nam,p),
        else:
            print ' %s %.3e' % (nam,norm(p)),
    print ''

    errTrainLast = errTrain.copy()
    PLast = P.copy()

    # Select and initialize learning rate rule
    if LRType == 'DecayRate':
        LR = DecayRate(errTrain,its,**LRParams)
    elif LRType == 'BoldDriver':
        LR = BoldDriver(errTrain,**LRParams)
    else:
        LR = LearningRate(errTrain,**LRParams)

    # Run until slope of validation error becomes positive, time runs out,
    # maximum iterations is reached, or learning rate falls to eps
    while ((slope < 0) or (its<extrapSteps)) and (time() < eschaton) and (its < maxIts) and (LR.lrate > eps):

        # For each training example, calculate gradient and update parameters
        for j in pr:
            y = Y[j]
            s = S[j,...]
            P +=  grad(y,s,P)*LR.lrate

        # Increment to next iteration
        its += 1

        # Calculate current training error and update learning rule
        R = Resp(S,P,resp)
        errTrain = cost(YR,R[pr])/errTrain0
        LR.update(errTrain)

        # If training error decreases
        if errTrain < errTrainLast:
            # Save new copies of last error and parameters
            errTrainLast = errTrain.copy()
            PLast = P.copy()

            # Calculate validation error
            errValid = cost(YV,R[pv])/errValid0
            errValidHist.append(errValid)

            # Calculate slope of the validation error
            if len(errValidHist) > extrapSteps:
                errValidHist = errValidHist[-extrapSteps:]
            x = ones((2,len(errValidHist)))
            x[1,:] = arange(len(errValidHist))
            slope = dot(inv(dot(x,x.T)),dot(x,array(errValidHist)))[1]

            # Save current parameters
            P.tofile(trainBestName)

            # Append errors to history files
            with open(errTrainName,'a') as f:
                errTrain.tofile(f)

            with open(errValidName,'a') as f:
                errValid.tofile(f)

            # If validation error has reached new minimum
            if errValid < errValidMin:

                # Update best value
                errValidMin = errValid

                # Copy parameters and save to parameter file
                PV = P.copy()
                PV.tofile(validBestName)

                # Output note of improvement
                errDown = errValidMin - errValid
                print '%u: New validation minimum %.5g, down %.3g' %(its,errValidMin,errDown)

            # Save current status
            with open(statusName,'w') as f:
                array(its).tofile(f)
                errValidMin.tofile(f)

            # Print status
            print '%u Values:' % (its,),
            for nam,p in zip(Pname,P):
                if p.size == 1:
                    print ' %s %.3e' % (nam,p),
                else:
                    print ' %s %.3e' % (nam,norm(p)),
            print ''
            print 'Slope %.3e' % (slope,)
        else:
            print 'Training error increased: learning rate too high'
            print 'New learning rate %.3e' % LR.lrate
            its -= 1
            P = PLast.copy()

    # If converged, delete status file
    if time() < eschaton and its < maxIts:
        remove(statusName)

    # Note that program has terminated successfully
    stdout.write('Time elapsed {0:.3f} hours\n'.format((time()-genesis)/3600.))
    stdout.write('Finished\n')
    stdout.flush()

