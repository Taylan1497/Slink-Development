import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 144

import pyomo
import pyomo.environ as pyo
#print(pyomo.__version__)
from pyomo.gdp import *
from pyomo.environ import value

import gekko
#print(gekko.__version__)
from gekko import GEKKO


linkRates = pd.read_hdf('../out/merged.h5', 'linkRates')

#sortedRates = linkRates[ ~linkRates.doubleDAQlpGBT ].sort_values(by='EvSize')
SizeSortedRates =  linkRates.sort_values(by='EvSize', ascending=True)

nGBTSingles = 66
nGBTDoubles = 6
maxSlinkRate = 300
nSlinks = 12 #336
nFPGA = 1 #28

#---------------------------------------------------

nGBTs = nGBTSingles + nGBTDoubles
maxGBTsPerSlink = 12
SlinkPerFPGA = 12

def master(mode, pyomo, gekko):
    #mode='LtoH' #{'HtoL': high to low, 'LtoH': low to high, 'Ht': highest, 'Lt': lowest}

    lowS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][:int(nGBTSingles/3)].to_dict()
    medS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ])/2)-int(nGBTSingles/6):int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ])/2)+int(nGBTSingles/6)].to_dict()
    highS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][-1*int(nGBTSingles/3):].to_dict()
    lowD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ]["EvSize"][:int(nGBTDoubles/3)].to_dict()
    medD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ]["EvSize"][int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ])/2)-int(nGBTDoubles/6):int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ])/2)+int(nGBTDoubles/6)].to_dict()
    highD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ]["EvSize"][-1*int(nGBTDoubles/3):].to_dict()

    if mode=='LtoH':
        medS.update(highS)
        lowS.update(medS)
        medD.update(highD)
        lowD.update(medD)
        singleGBTRates = lowS
        doubleGBTRates = lowD
    if mode=='HtoL':
        medS.update(lowS)
        highS.update(medS)
        medD.update(lowD)
        highD.update(medD)
        singleGBTRates = highS
        doubleGBTRates = highD
    if mode=='Lt':
        singleGBTRates = SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][:nGBTSingles].to_dict()
        doubleGBTRates = SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True)  ]["EvSize"][:nGBTDoubles].to_dict()
    if mode=='Ht':
        singleGBTRates = SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][-1*nGBTSingles:].to_dict()
        doubleGBTRates = SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True)  ]["EvSize"][-1*nGBTDoubles:].to_dict()
        
    #print(singleGBTRates)
    #print(doubleGBTRates)

    return [singleGBTRates, doubleGBTRates]

def Pyomo(multiplier, singleGBTRates, doubleGBTRates):
    model = pyo.ConcreteModel()
    #Single = list(SizeSortedRates[(singleGBTRates.index.map(str))
    #Double = list(SizeSortedRates[(doubleGBTRates.index.map(str))


    Single=[str(k) for k in singleGBTRates.keys()]
    Double = [str(k) for k in doubleGBTRates.keys()]
    Slink = [str(i) for i in range(1,nSlinks+1)]
    FPGA = [str(i) for i in range(1,nFPGA+1)]
    
    model.GKSingles = pyo.Var(Single, Slink, initialize = 0, within = pyo.Binary)
    model.GKDoubles = pyo.Var(Double, Slink, initialize = 0, within = pyo.Binary)
    model.GKSlinks = pyo.Var(Slink, FPGA, initialize = 0, within = pyo.Binary)

    def SetNumberofLinks1(model, i):
        return sum([ model.GKSingles[i, j] for j in Slink ]) == 1   
    def SetNumberofLinks2(model, i):
        return sum([ model.GKDoubles[i, j] for j in Slink ]) == 1    
    def SetNumberofLinks3(model, i):
        return sum([ model.GKSlinks[i, j] for j in FPGA ]) == 1  
    model.linkCut1 = pyo.Constraint(Single, rule = SetNumberofLinks1)
    model.linkCut2 = pyo.Constraint(Double, rule = SetNumberofLinks2)
    model.linkCut3 = pyo.Constraint(Slink, rule = SetNumberofLinks3)

    def maxGBTs(model, j):
        return sum([ model.GKSingles[i, j] for i in Single ]) + 2*sum([ model.GKDoubles[i, j] for i in Double ]) <= maxGBTsPerSlink
    def PerFPGA(model, j):   
        return sum([ model.GKSlinks[i, j] for i in Slink ]) == SlinkPerFPGA 
    model.maxlpGBT = pyo.Constraint(Slink, rule = maxGBTs)
    model.PerFPGA = pyo.Constraint(FPGA,  rule = PerFPGA)

    GKavSlinkRate = (( sum([i for i in singleGBTRates.values()])+sum([i for i in doubleGBTRates.values()]) )/nSlinks)
    maxSlinkRatePyomo = GKavSlinkRate*(multiplier)

    def RateCut(model, j):   
        return sum([ singleGBTRates[int(i)]*model.GKSingles[i, j] for i in Single ]) + sum([ doubleGBTRates[int(i)]*model.GKDoubles[i, j] for i in Double ]) <= maxSlinkRatePyomo        
    model.ratelimit = pyo.Constraint(Slink, rule = RateCut)


    def OBJrule(model):
        GKslinkRatesVariance=0
        for j in Slink:
            GKslinkRates=sum([ singleGBTRates[int(i)]*model.GKSingles[i, j] for i in Single ]) + sum([ doubleGBTRates[int(i)]*model.GKDoubles[i, j] for i in Double ]) 
            GKslinkRatesVariance +=  (GKslinkRates-GKavSlinkRate)**2  
        GKslinkRatesStDev = pyo.sqrt(GKslinkRatesVariance / (nSlinks-1))
        return GKslinkRatesStDev
    model.objective = pyo.Objective(rule=OBJrule, sense=pyo.minimize)

    pyo.SolverFactory('../PyomoSolvers/couenne-osx/couenne').solve(model, tee=False)
    #pyo.SolverFactory('../PyomoSolvers/apopt-master/apopt.py').solve(model, tee=False)

    #print(maxSlinkRatePyomo)
    model.objective.display()
    #model.pprint()
    
        
    return [GKavSlinkRate, value(model.objective)]

def Gekko(multiplier, singleGBTRates, doubleGBTRates, GekkoIter):
    m = GEKKO() # Initialize gekko
    m.options.SOLVER=1  # APOPT is an MINLP solver

    m.options.MAX_MEMORY=6

    # optional solver settings with APOPT
    # More details at https://apmonitor.com/wiki/index.php/Main/OptionApmSolver
    m.solver_options = ['minlp_maximum_iterations '+str(GekkoIter), \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol '+str(GekkoIter), \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 10', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.05', \
                        # covergence tolerance
                        'minlp_gap_tol 0.01',
                        #'minlp_print_level 10',
                        ]
    
    nGBTSingles=len(singleGBTRates.keys())
    nGBTDoubles=len(doubleGBTRates.keys())
    singleGBTRates=[i for i in singleGBTRates.values()]
    doubleGBTRates=[i for i in doubleGBTRates.values()]

    GKSingles = m.Array(
        m.Var, (nGBTSingles, nSlinks),
        value=np.random.choice([0, 1], p=[1-1/nSlinks, 1/nSlinks]), lb=0, ub=1, integer=True)
    GKDoubles = m.Array(
        m.Var, (nGBTDoubles, nSlinks),
        value=np.random.choice([0, 1], p=[1-1/nSlinks, 1/nSlinks]), lb=0, ub=1, integer=True)
    GKSlinks = m.Array(
        m.Var, (nSlinks, nFPGA),
        value=np.random.choice([0, 1], p=[1-1/nFPGA, 1/nFPGA]), lb=0, ub=1, integer=True)

    for i in range(nGBTSingles):
        m.Equation( sum([ GKSingles[i, j] for j in range(nSlinks) ]) == 1 )
    for i in range(nGBTDoubles):
        m.Equation( sum([ GKDoubles[i, j] for j in range(nSlinks) ]) == 1 )    
    for i in range(nSlinks):
        m.Equation( sum([ GKSlinks[i, j] for j in range(nFPGA) ]) == 1 ) 

    for j in range(nSlinks):
        m.Equation(
                sum([ GKSingles[i, j] for i in range(nGBTSingles) ])
            + 2*sum([ GKDoubles[i, j] for i in range(nGBTDoubles) ])
            <= maxGBTsPerSlink)
    for j in range(nFPGA):    
        m.Equation(
            sum([ GKSlinks[i, j] for i in range(nSlinks) ]) == SlinkPerFPGA)

    GKslinkRates = [
        m.Intermediate(
              sum([ singleGBTRates[i]*GKSingles[i, j] for i in range(nGBTSingles) ])
            + sum([ doubleGBTRates[i]*GKDoubles[i, j] for i in range(nGBTDoubles) ]),
            name=f'outRate_{j}')
        for j in range(nSlinks)]

    GKavSlinkRate = m.Const(
        ( sum(singleGBTRates)+sum(doubleGBTRates) ) / nSlinks)
	
    maxSlinkRateGekko = GKavSlinkRate.value*(multiplier)
	
    for j in range(nSlinks):    
        m.Equation(
           sum([ singleGBTRates[i]*GKSingles[i, j] for i in range(nGBTSingles) ]) + sum([ doubleGBTRates[i]*GKDoubles[i, j] for i in range(nGBTDoubles) ]) <= maxSlinkRateGekko)

    #print('Average SlinkRate: ',( sum(singleGBTRates)+sum(doubleGBTRates) ) / nSlinks)

    GKslinkRatesVariance = m.Intermediate(
        sum( (slinkRate-GKavSlinkRate)**2 for slinkRate in GKslinkRates) / (nSlinks-1))
    GKslinkRatesStDev = m.Intermediate( m.sqrt(GKslinkRatesVariance) )

    m.Obj(GKslinkRatesStDev)

    m.solve(disp=False)

    print(f"""
    Solution found with
     - Average rate of Slinks = {GKavSlinkRate.value}
     - StdDev of Slinks rates = {GKslinkRatesStDev.value[0]}
    """)
    for j in range(nSlinks):  
        #print(GKslinkRates)
        print(GKslinkRates[j].value[0])
        
    return [GKavSlinkRate.value, GKslinkRatesStDev.value[0]]


