#!/usr/bin/env python
# coding: utf-8

# # Assigning the lpGBTs to the Slinks
# 
# Main boundary conditions:
# - Done per 120-degree sector.
# - Each 120-degree sector has 28 FPGAs.
# - Each FPGA has either:
#   - 12× 16G Slink outputs => 336 Slink outputs, or
#   - 8× 25G Slink outputs => 224 Slink outputs.
# 
# One idea:
# - Straddling ECONDs imply that an lpGBT pair is the quantum of the system.
# - Single lpGBTs need to be paired:
#   - Sort their rates and then take largest and smallest in pairs.
# - Pairs of lpGBTs are then assigned to individual Slinks:
#   - Max of 8 lpGBT pairs per Slink.
# 

# In[1]:


import cloudpickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import sys 
mpl.rcParams['figure.dpi'] = 144
import numpy as np
import pyomo
import pyomo.environ as pyo
print(pyomo.__version__)
from pyomo.gdp import *
from pyomo.environ import value
import operator


# In[2]:


linkRates = pd.read_hdf('out/merged.h5', 'linkRates')



# In[4]:


n_doubles = len(linkRates[ linkRates.doubleDAQlpGBT ])
n_singles = len(linkRates[ ~linkRates.doubleDAQlpGBT ])
n_links = n_singles + 2*n_doubles
print(f"""
There is a total of {n_links} lpGBT links:
 - {n_singles} singles, and
 - {n_doubles} doubles (with one ECOND straddling the pair)
""")


# In[5]:


sortedRates = linkRates[ ~linkRates.doubleDAQlpGBT ].sort_values(by='EvSize')


# In[6]:


#SizeSortedRates =  linkRates
SizeSortedRates =  linkRates.sort_values(by='EvSize', ascending=True)
SizeSortedRates


# In[7]:


assert( len(sortedRates)%2 == 0 )


# Since there are singles with larger rates than some doubles, the naive pairing of singles need not be a good idea.

# In[8]:


linkRates['nDAQlpGBT'] = linkRates.doubleDAQlpGBT.apply(lambda d: 2 if d else 1)


# # A playground for optimization functions
# 
# This problem cannot be solved with linear-programming techniques.
# 
# Since the assignment of a lpGBT to a Slink is a boolean variable, we're in the domain of [mixed-integer non-linear programming (MINLP)](https://www.google.com/search?q=python+MINLP).
# 
# I found two python libraries that model such problems: `GEKKO` and `pyomo`.
# 
# Some other resources:
# - http://plato.asu.edu/sub/tools.html
# - https://www.zanaducloud.com/7935420D-38BB-47B4-AD1C-ADEF76CA6884
# - https://www.localsolver.com/download.html

# ## Setting up the toy problem

# In[9]:


nFPGA = 28 #max=28
#runNumber = 3 #max=4
splitNumber = 16

warmStart=False
hardStart=True

#---------------------------------------------------

nGBTSingles = int((1288*nFPGA)/28)
#nGBTSingles = len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ])
nGBTDoubles = int((110*nFPGA)/28)
#nGBTDoubles = len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ])
#maxSlinkRate = 300
nSlinks = nFPGA*12  #336

#---------------------------------------------------

nGBTs = nGBTSingles + nGBTDoubles
maxGBTsPerSlink = 12
SlinkPerFPGA = 12


# In[10]:


print(f"""
Toy problem parameters:
- {nGBTs} lpGBT inputs.
  - {nGBTSingles} lpGBT singles, and
  - {nGBTDoubles} lpGBT doubles.
- {nSlinks} Slink outputs.
  - Up to {maxGBTsPerSlink} lpGBTs served by each SLink.
""")
print(f"""
 - {nFPGA} FPGAs and
 - {SlinkPerFPGA} Slinks served by each FPGA.
""")


# In[11]:


def split_dict_equally(input_dict, chunks=2):
    "Splits dict by keys. Returns a list of dictionaries."
    # prep with empty dicts
    return_list = [dict() for idx in range(chunks)]
    idx = 0
    for k,v in input_dict.items():
        return_list[idx][k] = v
        if idx < chunks-1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list


# In[12]:


mode='HtoL' #{'HtoL': high to low, 'LtoH': low to high, 'Ht': highest, 'Lt': lowest}

lowS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ].set_index('DAQlpGBT')["EvSize"][:int(nGBTSingles/3)].to_dict()
medS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ].set_index('DAQlpGBT')["EvSize"][int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ])/2)-int(nGBTSingles/6):int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ])/2)+int(nGBTSingles/6)].to_dict()
if nFPGA==28:
    medS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ].set_index('DAQlpGBT')["EvSize"][int(nGBTSingles/3):-1*int(nGBTSingles/3)].to_dict()
highS=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ].set_index('DAQlpGBT')["EvSize"][-1*int(nGBTSingles/3):].to_dict()

lowD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ].set_index('DAQlpGBT')["EvSize"][:int(nGBTDoubles/3)].to_dict()
medD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ].set_index('DAQlpGBT')["EvSize"][int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ])/2)-int(nGBTDoubles/6):int(len(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ])/2)+int(nGBTDoubles/6)].to_dict()
if nFPGA==28:
    medD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ].set_index('DAQlpGBT')["EvSize"][int(nGBTDoubles/3):-1*int(nGBTDoubles/3)].to_dict()
highD=SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True) ].set_index('DAQlpGBT')["EvSize"][-1*int(nGBTDoubles/3):].to_dict()

#------------------------------------------------------------

lowS_1={}
medS_1={}
highS_1=highS

lowD_1=lowD
medD_1=medD
highD_1=highD

    
#------------------------------------------------------------

if mode=='LtoH':
    medS_1.update(highS_1)
    lowS_1.update(medS_1)
    
    medD_1.update(highD_1)
    lowD_1.update(medD_1)
    
    singleGBTRates_1 = lowS_1
    doubleGBTRates_1 = lowD_1
if mode=='HtoL':
    medS_1.update(lowS_1)
    highS_1.update(medS_1)
    
    medD_1.update(lowD_1)
    highD_1.update(medD_1)
    
    singleGBTRates_1 = highS_1
    doubleGBTRates_1 = highD_1
if mode=='Lt':
    SizeSortedRates =  linkRates.sort_values(by='EvSize', ascending=True)
    singleGBTRates_1 = split_dict_equally(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][:nGBTSingles].to_dict())[1]
    doubleGBTRates_1 = split_dict_equally(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True)  ]["EvSize"][:nGBTDoubles].to_dict())[1]
if mode=='Ht':
    SizeSortedRates =  linkRates.sort_values(by='EvSize', ascending=False)
    singleGBTRates_1 = split_dict_equally(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== False) ]["EvSize"][:nGBTSingles].to_dict())[1]
    doubleGBTRates_1 = split_dict_equally(SizeSortedRates[(SizeSortedRates["doubleDAQlpGBT"]== True)  ]["EvSize"][:nGBTDoubles].to_dict())[1]

#print(len(singleGBTRates_1))
#print(singleGBTRates_1)
#print(len(doubleGBTRates_1))
#print(doubleGBTRates_1)


# In[13]:


totalRate = ( sum([i for i in singleGBTRates_1.values()])+sum([i for i in doubleGBTRates_1.values()]) )


# In[14]:


print(f"""
Total lpGBT rate: {totalRate}
Average lpGBT rate: {totalRate /len(doubleGBTRates_1)+len(doubleGBTRates_1)}
Average Slink rate: {totalRate / nSlinks}
""")


# In[15]:


#maxSlinkRatePyomo = (totalRate / nSlinks)*1.8
maxSlinkRatePyomo = 600
#maxSlinkRatePyomo = (totalRate / (len(doubleGBTRates_1)+len(singleGBTRates_1)))*1.5
maxSlinkRateGekko = (totalRate / nSlinks)*1.4
print("maxSlinkRatePyomo: ",maxSlinkRatePyomo)
#print("maxSlinkRateGekko: ",maxSlinkRateGekko)


# # Assigning the lpGBTs to the Slinks

# In[16]:


avSlinkEvSize = totalRate/nSlinks


# In[17]:


f, ax = plt.subplots(figsize=(7, 3))
sns.distplot(
    [i for i in singleGBTRates_1.values()],
    kde=False,
    rug=True,
    label='Single lpGBT'
    )
plt.annotate('Average', xy=(avSlinkEvSize,0), xytext=(avSlinkEvSize, 30),  ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1))
sns.distplot(
    [i for i in doubleGBTRates_1.values()],
    kde=False,
    rug=True,
    label='Double lpGBT'
    )
plt.legend()


# In[18]:


from dataclasses import dataclass
@dataclass
class Slink:
    totalEvSize: float
    linkList: list

from operator import attrgetter
def assign_link(link, Slinks):
    Slinks.sort(key=attrgetter('totalEvSize'))
    target = Slinks[0]
    target.linkList.append(link[0])
    target.totalEvSize += link[1]
    return Slinks

Slinks = [Slink(totalEvSize=0, linkList=[]) for _ in range(nSlinks)]

#for link in sortedLinkRates.itertuples(name='Link', index=False):
#    Slinks = assign_link(link, Slinks)

lpGBTs={}
lpGBTs.update(doubleGBTRates_1)
lpGBTs.update(singleGBTRates_1)
lpGBTs={k: v for k, v in sorted(lpGBTs.items(), key=lambda item: item[1],reverse=True)}
for link in lpGBTs.items():
    Slinks = assign_link(link, Slinks)


# In[19]:


SlinkTotals = pd.DataFrame(data=[ Slink.totalEvSize for Slink in Slinks ], columns=['EvSize'])
#print(SlinkTotals)


# In[20]:


f, ax = plt.subplots(figsize=(7, 3))
sns.distplot(
    SlinkTotals['EvSize'],
    kde=False,
    rug=True,
    color = 'g',
    label='Slinks'
    )
plt.annotate('Average', xy=(avSlinkEvSize,0), xytext=(avSlinkEvSize, 30),  ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=1))
plt.legend()


# In[21]:


#SlinkTotals.describe()


# In[22]:


#print(f'RMS/mean = {float(SlinkTotals.std()/SlinkTotals.mean()):.2%}')


# ## Pyomo Start

# Pyomo solver options are set in the file "../apopt-master/apopt.py".

# In[23]:


model = pyo.ConcreteModel()


# In[24]:


Single1=[str(k) for k in singleGBTRates_1.keys()]
Double1 = [str(k) for k in doubleGBTRates_1.keys()]
Slink = [str(i) for i in range(1,nSlinks+1)]
FPGA = [str(i) for i in range(1,nFPGA+1)]
#print(Single1)
#print(Double1)
#print(Slink)
#print(FPGA)


# In[25]:


model.GKSingles1 = pyo.Var(Single1, Slink, initialize = 0, within = pyo.Binary)
model.GKDoubles1 = pyo.Var(Double1, Slink, initialize = 0, within = pyo.Binary)
model.GKSlinks = pyo.Var(Slink, FPGA, initialize = 0, within = pyo.Binary)


# In[26]:


if warmStart==True or hardStart==True:
    i=1
    for slink in Slinks:
        #print(slink.linkList)
        for link in slink.linkList:
            #print(link)
            if str(link) in Single1:
                model.GKSingles1[str(link),str(i)]=1
            elif str(link) in Double1:
                model.GKDoubles1[str(link),str(i)]=1
        i+=1


# In[27]:


def Sort(sub_li): 
    l = len(sub_li) 
    for i in range(0, l): 
        for j in range(0, l-i-1): 
            if (sub_li[j][1] > sub_li[j + 1][1]): 
                tempo = sub_li[j] 
                sub_li[j]= sub_li[j + 1] 
                sub_li[j + 1]= tempo 
    return sub_li 


# In[28]:


def swapData(L1,L2,L3):
    GK={}
    for j in Slink:
        GK[j]=[0,[]]
        for ilist,var,arr in zip(L1,L2,L3):
            for i in ilist:
                if var[i, j].value>0.1:
                    GK[j][0] += arr[int(i)]*var[i,j].value
                    GK[j][1].append([int(i),arr[int(i)]])
    return GK


# def SwapModule(slinks,SlinkAve,L1,L2):
#     ss=[i for i in {k: v for k, v in sorted(slinks.items(), key=lambda item: item[1],reverse=True)}.items()]
#     for i in ss:
#         if len(i[1][1])!=1:
#             slinkHigh=i
#             break
#     lpH,lpHR=Sort(slinkHigh[1][1])[0][0],Sort(slinkHigh[1][1])[0][1]
#  
#     flag=0
#     for i in ss:
#         for j in i[1][1]:
#             dif=lpHR-j[1]
#             if dif>0 and i[1][0]+dif<slinkHigh[1][0] and slinkHigh[1][0]-dif>ss[-1][1][0]:
#                 flag=1
#                 lpL=j[0]
#                 lpLR=j[1]
#                 slinkLow=i
#                 
#     if flag==1:
#         for ilist,var in zip(L1,L2):
#             if str(lpH) in ilist:
#                 var[str(lpH),slinkHigh[0]].value=0
#                 var[str(lpH),slinkLow[0]].value=1
#             elif str(lpL) in ilist:
#                 var[str(lpL),slinkLow[0]].value=0
#                 var[str(lpL),slinkHigh[0]].value=1
#             else:
#                 continue

# In[29]:


def SwapModule(SlinkAve,L1,L2,L3):
    ss=[i for i in {k: v for k, v in sorted(swapData(L1,L2,L3).items(), key=lambda item: item[1],reverse=True)}.items()]
    for h in ss:
        if len(h[1][1])!=1:
            slinkHigh=h
            #break
        else: continue
        lpH,lpHR=Sort(slinkHigh[1][1])[0][0],Sort(slinkHigh[1][1])[0][1]
        tempH=abs(slinkHigh[1][0]-SlinkAve)
        
        flag=0
        tempLG=100000000
        for l in ss:
            tempL=abs(l[1][0]-SlinkAve)
            for j in l[1][1]:
                dif=lpHR-j[1]
                if dif>0 and l[1][0]+dif<slinkHigh[1][0] and slinkHigh[1][0]-dif>ss[-1][1][0]:
                    if abs((l[1][0]+dif)-SlinkAve)<tempL and abs((slinkHigh[1][0]-dif)-SlinkAve)<tempH:
                        if abs((l[1][0]+dif)-SlinkAve)<tempLG:
                            flag=1
                            tempLG=abs((l[1][0]+dif)-SlinkAve)
                            lpL=j[0]
                            lpLR=j[1]
                            slinkLow=l
        
                
        if flag==1:
            for ilist,var in zip(L1,L2):
                if str(lpH) in ilist:
                    var[str(lpH),slinkHigh[0]].value=0
                    var[str(lpH),slinkLow[0]].value=1
                elif str(lpL) in ilist:
                    var[str(lpL),slinkLow[0]].value=0
                    var[str(lpL),slinkHigh[0]].value=1
                else:
                    continue
            ss=[i for i in {k: v for k, v in sorted(swapData(L1,L2,L3).items(), key=lambda item: item[1],reverse=True)}.items()]


# In[30]:


#model.pprint()


# for j in Slink:
#     for i in Single1:
#         if model.GKSingles1[i,j].value==1:
#             print(i,j)
#             print(model.GKSingles1[i,j].value)
#     for i in Double1:
#         if model.GKDoubles1[i,j].value==1:
#             print(i,j)
#             print(model.GKDoubles1[i,j].value)

# In[31]:


def SetNumberofLinks1(model, i):
    return sum([ model.GKSingles1[i, j] for j in Slink ]) == 1
    
def SetNumberofLinks2(model, i):
    return sum([ model.GKDoubles1[i, j] for j in Slink ]) == 1
    
def SetNumberofLinks3(model, i):
    return sum([ model.GKSlinks[i, j] for j in FPGA ]) == 1
    
model.linkCutS1 = pyo.Constraint(Single1, rule = SetNumberofLinks1)
model.linkCutD1 = pyo.Constraint(Double1, rule = SetNumberofLinks2)
model.linkCutSK = pyo.Constraint(Slink, rule = SetNumberofLinks3)


# In[32]:


def maxGBTs(model, j):
    return sum([ model.GKSingles1[i, j] for i in Single1 ]) + 2*sum([ model.GKDoubles1[i, j] for i in Double1 ]) <= maxGBTsPerSlink

def PerFPGA(model, j):   
    return sum([ model.GKSlinks[i, j] for i in Slink ]) == SlinkPerFPGA 

model.maxlpGBT = pyo.Constraint(Slink, rule = maxGBTs)
model.PerFPGA = pyo.Constraint(FPGA,  rule = PerFPGA)


# In[33]:


def RateCut(model, j):   
    return sum([ singleGBTRates_1[int(i)]*model.GKSingles1[i, j] for i in Single1 ]) + sum([ doubleGBTRates_1[int(i)]*model.GKDoubles1[i, j] for i in Double1 ]) <= maxSlinkRatePyomo 
        
model.ratelimit = pyo.Constraint(Slink, rule = RateCut)


# In[34]:


GKavSlinkRate = (totalRate/nSlinks)
#print(GKavSlinkRate)


# def OBJrule(model):
#     GKslinkRatesVariance=0
#     for j in Slink:
#         GKslinkRates=sum([ singleGBTRates[int(i)]*model.GKSingles[i, j] for i in Single ]) + sum([ doubleGBTRates[int(i)]*model.GKDoubles[i, j] for i in Double ]) 
#         GKslinkRatesVariance +=  (GKslinkRates-GKavSlinkRate)**2  
#     GKslinkRatesStDev = pyo.sqrt(GKslinkRatesVariance / (nSlinks-1))
#     return GKslinkRatesStDev
# model.objective = pyo.Objective(rule=OBJrule, sense=pyo.minimize)

# In[35]:


def OBJrule(model):
    GKslinkRatesVariance=0
    for j in Slink:
        GKslinkRates=sum([ singleGBTRates_1[int(i)]*model.GKSingles1[i, j] for i in Single1 ]) + sum([ doubleGBTRates_1[int(i)]*model.GKDoubles1[i, j] for i in Double1 ]) 
        GKslinkRatesVariance +=  abs(GKslinkRates-GKavSlinkRate)  
    GKslinkRatesStDev = (GKslinkRatesVariance / (nSlinks-1))
    return GKslinkRatesStDev
model.objective = pyo.Objective(rule=OBJrule, sense=pyo.minimize)


# In[36]:


#model.pprint()


# In[37]:


if hardStart==False:
    #pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)
    pyo.SolverFactory('./PyomoSolvers/couenne-osx/couenne').solve(model, tee=False)
    #pyo.SolverFactory('./PyomoSolvers/SCIPOptSuite-7.0.1-Darwin/bin/scip').solve(model, tee=True)
    #pyo.SolverFactory('./PyomoSolvers/apopt-master/apopt.py').solve(model, tee=True) 

model.GKSingles1.fix()
model.GKDoubles1.fix()

with open('./ScriptRuns/'+str(nFPGA)+'FPGA_run1.pkl', mode='wb+') as file:
    cloudpickle.dump(model, file)

# with open('./Runs/'+str(nFPGA)+'FPGA_run3.pkl', mode='rb') as file:
#    model = cloudpickle.load(file)

# In[38]:


print("Average Slink Rate:", GKavSlinkRate)
print("Std: ", value(model.objective))
model.objective.display()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Run 2

# In[42]:


for splitDict,splitDictName,dictNo in zip([lowS],['lows'],['3']):
#for splitDict,splitDictName,dictNo in zip([medS, lowS],['mediums','lows'],['2','3']):
    #for partNumber in range(splitNumber):
    for partNumber in [6,7,8]:
        print('-----------------------------------------------')
        # In[43]:
        if partNumber==0 and dictNo=='2':
            with open('./ScriptRuns/'+str(nFPGA)+'FPGA_run1.pkl', mode='rb') as file:
                model = cloudpickle.load(file)
        elif partNumber==0 and dictNo=='3':
        	with open('./ScriptRuns/mediums'+str(nFPGA)+'FPGA_run2_part'+str(splitNumber-1)+'.pkl', mode='rb') as file:
        		model = cloudpickle.load(file)
        else:
            with open('./ScriptRuns/'+str(splitDictName)+str(nFPGA)+'FPGA_run2_part'+str(partNumber-1)+'.pkl', mode='rb') as file:
                model = cloudpickle.load(file)


        # In[44]:

        
        singleGBTRates_2 = split_dict_equally(splitDict,splitNumber)[partNumber]

        print('len(singleGBTRates_2): ',len(singleGBTRates_2))
        #print(singleGBTRates_2)


        Single2=[str(k) for k in singleGBTRates_2.keys()]


        # In[46]:
        print('Doing part: ',partNumber)
        name='GKSingles'+dictNo+'_'+str(partNumber)
        model.add_component(name, pyo.Var(Single2, Slink, initialize = 0, within = pyo.Binary))
    

        #for v in model.component_data_objects(pyo.Var, active=True):
        #    print(v, pyo.value(v))  # doctest: +SKIP
        #model.pprint()
        Vars= {}
        for v in model.component_objects(pyo.Var, active=True):
            #print("Variable",v) 
            lpList=[]
            if v.name.startswith('GKSingles'):
                for index in v:
                    #print ("   ",index, pyo.value(v[index]))
                    if index[0] not in lpList:
                        lpList.append(index[0])
                    #print(index[0])
                Vars[v.name]=[lpList,v] # doctest: +SKIP

        #print(Vars,'\n')

        singleGBTRates_2_ALL={}
        for k,v in Vars.items():
            if not k.startswith('GKSingles1'):
                tempDict=medS if k.startswith('GKSingles2') else lowS
                for l in v[0]:
                    singleGBTRates_2_ALL[int(l)]=tempDict[int(l)]

        #print(singleGBTRates_2_ALL)
        Single2_ALL = [str(k) for k in singleGBTRates_2_ALL.keys()]

        totalRate = ( sum([i for i in singleGBTRates_1.values()])+sum([i for i in singleGBTRates_2_ALL.values()])+sum([i for i in doubleGBTRates_1.values()]) )
        GKavSlinkRate = (totalRate/nSlinks)
        #maxSlinkRatePyomo = (totalRate / nSlinks)*1.8
        maxSlinkRatePyomo =600
        print('GKavSlinkRate: ', GKavSlinkRate)
        print('maxSlinkRatePyomo: ', maxSlinkRatePyomo)



        #print(Vars)
        if partNumber!=0 or dictNo=='3':
            model.del_component(model.linkCutS2)
            model.del_component(model.linkCutS2_index)
        def SetNumberofLinks12(model, i):
            name='GKSingles'+dictNo+'_'+str(partNumber)
            return sum( [Vars[name][1][i, j] for j in Slink ]) == 1
        #def SetNumberofLinks22(model, i):
        #    return sum([ model.GKDoubles2[i, j] for j in Slink ]) == 1
        model.linkCutS2 = pyo.Constraint(Single2, rule = SetNumberofLinks12)
        #model.linkCutD2 = pyo.Constraint(Double2, rule = SetNumberofLinks22)

        model.del_component(model.maxlpGBT)
        model.del_component(model.maxlpGBT_index)
        def maxGBTs(model, j):
            GKSingles2=0
            dNL = ['2','3'] if dictNo=='3' else ['2']
            for dN in dNL:
            	for pNo in range(partNumber+1):
                	name='GKSingles'+dN+'_'+str(pNo)
                	GKSingles2 += sum( [ Vars[name][1][i, j] for i in Vars[name][0]])
            return sum([ model.GKSingles1[i, j].value for i in Single1]) + GKSingles2 + 2*sum([ model.GKDoubles1[i, j].value  for i in Double1 ])  <= maxGBTsPerSlink
        model.maxlpGBT = pyo.Constraint(Slink, rule = maxGBTs)

        model.del_component(model.ratelimit)
        model.del_component(model.ratelimit_index)
        def RateCut(model, j):   
            GKSingles2=0
            dNL = ['2','3'] if dictNo=='3' else ['2']
            for dN in dNL:
            	for pNo in range(partNumber+1):
                	name='GKSingles'+dN+'_'+str(pNo)
                	GKSingles2 += sum( [singleGBTRates_2_ALL[int(i)]*Vars[name][1][i, j] for i in Vars[name][0] ])
            return (1 , sum([ singleGBTRates_1[int(i)]*model.GKSingles1[i, j] for i in Single1])+ GKSingles2  + sum([ doubleGBTRates_1[int(i)]*model.GKDoubles1[i, j] for i in Double1])  , maxSlinkRatePyomo ) 
        model.ratelimit = pyo.Constraint(Slink, rule = RateCut)

        model.del_component(model.objective)
        def OBJrule(model):
            GKslinkRatesVariance=0
            for j in Slink:
                GKSingles2=0
                c=0
                dNL = ['2','3'] if dictNo=='3' else ['2']
                for dN in dNL:
                    for pNo in range(partNumber+1):
                        c+=1
                        name='GKSingles'+dN+'_'+str(pNo)
                        GKSingles2 += sum( [singleGBTRates_2_ALL[int(i)]*Vars[name][1][i, j] for i in Vars[name][0] ])
                GKslinkRates=sum([ singleGBTRates_1[int(i)]*model.GKSingles1[i, j] for i in Single1])+ GKSingles2  + sum([ doubleGBTRates_1[int(i)]*model.GKDoubles1[i, j] for i in Double1])
                GKslinkRatesVariance +=  abs(GKslinkRates-GKavSlinkRate)
            GKslinkRatesStDev = (GKslinkRatesVariance / (nSlinks-1))
            return GKslinkRatesStDev
        model.objective = pyo.Objective(rule=OBJrule, sense=pyo.minimize)


        # In[47]:


        #model.pprint()


        # In[49]:


        #pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)
        pyo.SolverFactory('./PyomoSolvers/couenne-osx/couenne').solve(model, tee=False)
        #pyo.SolverFactory('./PyomoSolvers/SCIPOptSuite-7.0.1-Darwin/bin/scip').solve(model, tee=True)
        #pyo.SolverFactory('./PyomoSolvers/apopt-master/apopt.py').solve(model, tee=True)


        # In[50]:
        name='GKSingles'+dictNo+'_'+str(partNumber)
        for i in Vars[name][0]:
            sumLink=0
            for j in Slink:
                sumLink+=Vars[name][1][i,j].value
            if sumLink!=1:
                print('Error! Not all lpGBTs are linked!')
                sys.exit()


        # In[51]:


        Vars[name][1].fix()


        # In[52]:


        with open('./ScriptRuns/'+str(splitDictName)+str(nFPGA)+'FPGA_run2_part'+str(partNumber)+'.pkl', mode='wb+') as file:
            cloudpickle.dump(model, file)


        # In[53]:

        print("Average Slink Rate:", GKavSlinkRate)
        print("Std: ", value(model.objective))
        model.objective.display()

    # def swapData():
    #     GK={}
    #     for j in Slink:
    #         GK[j]=[0,[]]
    #         for ilist,var,arr in zip([Single1,Single2,Double1],[model.GKSingles1,model.GKSingles2,model.GKDoubles1],[singleGBTRates_1,singleGBTRates_2,doubleGBTRates_1]):
    #             for i in ilist:
    #                 if var[i, j].value>0.1:
    #                     GK[j][0] += arr[int(i)]*var[i,j].value
    #                     GK[j][1].append([int(i),arr[int(i)]])
    #     return GK
    # temp=1000000000
    # flag=0
    # #while runNumber==2:
    # while True:
    #     L1=[Single1,Single2,Double1]
    #     L2=[model.GKSingles1,model.GKSingles2,model.GKDoubles1]
    #     L3=[singleGBTRates_1,singleGBTRates_2,doubleGBTRates_1]
    #     SwapModule(swapData(),GKavSlinkRate,L1,L2,L3)
    #     std=value(model.objective)
    #     print("Std: ", std)
    #     if std<temp:
    #         temp=std
    #         flag=1
    #     elif std==temp:
    #         if flag==3:
    #             break
    #         flag+=1
    #         
    # for i in swapData().keys():
    #     print(i,' ', swapData()[i][0])
    # SlinkList=[i for i in {k: v for k, v in sorted(swapData().items(), key=lambda item: item[1],reverse=True)}.items()]
    # print(SlinkList[0])
    # print(SlinkList[-1])

# In[54]:


GK={}
for j in Slink:
    GKSingles2_3=0
    for dictNo in ['2','3']:
        for pNo in range(splitNumber):
            name='GKSingles'+dictNo+'_'+str(pNo)
            GKSingles2_3 += sum( [singleGBTRates_2_ALL[int(i)]*Vars[name][1][i, j].value for i in Vars[name][0] ])
    GK[j] = GKSingles2_3 + sum([ singleGBTRates_1[int(i)]*model.GKSingles1[i, j].value for i in Single1 ]) + sum([ doubleGBTRates_1[int(i)]*model.GKDoubles1[i, j].value for i in Double1 ])  
#print(GK)
#for j in Slink:
    #print(GK[j])
slink_rate_list=[]
for j in GK:
    slink_rate_list.append(GK[j])
    #slink_rate_list
slink_array=[]
for i in Slink:
    list=[]
    for j in FPGA:
        if 0.9<=pyo.value(model.GKSlinks[i,j])<=1.00001:
            list.append(1)
        else: #pyo.value(model.GKSlinks[i,j])==0:
            list.append(0)
    slink_array.append(list)
Slinks=np.array(slink_array)
Slinks_Rates=np.array(slink_rate_list)
hml_slink=(Slinks.T*Slinks_Rates).T
#hml_slink
f, ax = plt.subplots(figsize=(7, 3))
sns.distplot(
    slink_rate_list,
    kde=False,
    label='Slinks',
    color='g',
    rug=True,
    )
plt.legend()
ax.set(xlabel='Slink av. rate', ylabel='Number of Slinks')
plt.axvline(linewidth=1, color='r',x=totalRate / nSlinks)
#plt.savefig("Slink.png")
###FPGA PART####
GK2={}
for j in FPGA:
    GK2[j] = sum([ slink_rate_list[int(i)-1]*model.GKSlinks[i, j].value for i in Slink])   
#print(GK)
#for j in FPGA:
    #print(GK2[j])
fpga_rate_list=[]
for j in GK2:
    fpga_rate_list.append(GK2[j])
#fpga_rate_list
f, ax = plt.subplots(figsize=(7, 3))
sns.distplot(
    fpga_rate_list,
    kde=False,
    label='FPGAs',
    color='b',
    rug=True,
    )
plt.legend()
ax.set(xlabel='FPGA av. rate', ylabel='Number of FPGAs')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
#FPGAs = ["1","2"]  
FPGAs=[str(i) for i in range(1,nFPGA+1)]
Rates = fpga_rate_list
ax.set_ylabel('Lpgbt Rates')
ax.set_xlabel("FPGAs")
ax.bar(FPGAs,Rates)
plt.show()
###FPGA VS SLİNK
(wd, ht) = (7, 6)

vmax = max(Slinks_Rates)
f, ax = plt.subplots(figsize=(wd, ht))
sns.heatmap (
        (Slinks.T*Slinks_Rates).T,
        cmap='Greens', cbar_kws={'label': 'Av. rate'},
        vmin=0, vmax=vmax
    )
ax.set(xlabel='FPGA', ylabel='Slink')

# In[53]:


'''
# In[53]:

#sns.set(rc={'figure.figsize':(11.7,8.27)})
#plt.figure(figsize=(16,6))
single_array=[]
single_list=[Single1,Single2]
for i in single_list:
    for k in i:
        for j in Slink:
            list=[0]*nSlinks
            if k in Single1:
                if 0.9<=pyo.value(model.GKSingles1[k,j])<=1.0000009:
                    a=int(j)
                    list[a-1]=1           
                    single_array.append(list)
            if k in Single2:
                if 0.9<=pyo.value(model.GKSingles2[k,j])<=1.0000009:
                    a=int(j)
                    list[a-1]=1           
                    single_array.append(list)
singles=np.array(single_array)
s=[]
for i in singleGBTRates_1.values():
    s.append(i)
for i in singleGBTRates_2.values():
    s.append(i)
singleGBTRatesMAIN=np.array(s)
hml_single=(singles.T*singleGBTRatesMAIN).T
#hml_single[:,0]
double_array=[]
for i in Double1:
    for j in Slink:
        list=[0]*nSlinks
        if 0.9<=pyo.value(model.GKDoubles1[i,j])<=1.00009:#or ==singleGBTRates[i]
            a=int(j)# 1 en 12 ye kadar bir sayı
            list[a-1]=1
            #print(list)
            double_array.append(list)
d=[]
for i in doubleGBTRates_1.values():
    d.append(i)
doubleGBTRatesMAIN=np.array(d)
doubles=np.array(double_array)
hml_double=(doubles.T*doubleGBTRatesMAIN).T
#hml_double[:0]

#### SCATTER PLOT PART
hml_temp_single=[]
for j in range(nSlinks):
    a=[]
    for i in hml_single[:,j]:
        if i !=0:
            a.append(i)
    #b=np.array(a)
    hml_temp_single.append(a)
hml2_single=np.array([hml_temp_single])

hml_temp_double=[]
for j in range(nSlinks):
    a=[]
    for i in hml_double[:,j]:
        if i !=0:
            a.append(i)
    #b=np.array(a)
    hml_temp_double.append(a)
hml2_double=np.array([hml_temp_double])

df_tot=pd.DataFrame({"Slinks":[],"Single Lpgbt Rates":[],"Double Lpgbt Rates":[]})
for i in range(nSlinks):
    c={"Slinks":[str(i)]*len(hml_temp_single[i]),"Single Lpgbt Rates":hml_temp_single[i],"Double Lpgbt Rates":hml_temp_double[i]}
    df2_tot=pd.DataFrame({ key:pd.Series(value) for key, value in c.items() })

    df_tot=pd.concat([df_tot,df2_tot],ignore_index=True)
    #df.append(df2,ignore_index=True)
pd.set_option('display.max_rows', None)

df_double2=pd.DataFrame({"Slinks":[],"Lpgbt Rates":[],"link variable":[]})
for i in range(nSlinks):
    c={"Slinks":[str(i)]*len(hml_temp_double[i]),"Lpgbt Rates":hml_temp_double[i],"link variable":["double"]*len(hml_temp_double[i]),"total rate":slink_rate_list[i]}
    df2_single=pd.DataFrame(c)
    df2_double=pd.DataFrame(c)
    df_double2=pd.concat([df_double2,df2_double],ignore_index=True)
    #df.append(df2,ignore_index=True)
pd.set_option('display.max_rows', None)

df_single2=pd.DataFrame({"Slinks":[],"Lpgbt Rates":[],"link variable":[]})
for i in range(nSlinks):
    c={"Slinks":[str(i)]*len(hml_temp_single[i]),"Lpgbt Rates":hml_temp_single[i],"link variable":["single"]*len(hml_temp_single[i]),"total rate":(slink_rate_list[i])}
    df2_single=pd.DataFrame(c)
    df_single2=pd.concat([df_single2,df2_single],ignore_index=True)
    
df_together=pd.concat([df_single2,df_double2], axis=0)
sns_scatter=sns.scatterplot(data=df_together, x="Slinks", y="Lpgbt Rates", hue="link variable")
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#fig=sns_scatter.get_figure()
#fig.savefig("scatterplot.png")
   


###SİNGLE PART

(wd, ht) = (7, 6)
vmax = max(singleGBTRatesMAIN)
f, ax = plt.subplots(figsize=(wd, ht))
sns.heatmap(
    (singles.T * singleGBTRatesMAIN).T,
    cmap='Blues', cbar_kws={'label': 'Av. rate'},
    vmin=0, vmax=vmax
)
ax.set(xlabel='Slink', ylabel='lpGBT single')

### DOUBLE PART

f, ax = plt.subplots(figsize=(wd, ht * (nGBTDoubles/nGBTSingles)))
vmax = max(doubleGBTRatesMAIN)
sns.heatmap(
    ((doubles.T) * doubleGBTRatesMAIN).T,
    cmap='Oranges', cbar_kws={'label': 'Av. rate'},
    vmin=0, vmax=vmax
)
ax.set(xlabel='Slink', ylabel='lpGBT double')

### DATAFRAME PART

def create_dataframe():
    hml_double=(doubles.T*doubleGBTRatesMAIN).T
    hml_single=(singles.T * singleGBTRatesMAIN).T
    a=[]
    for j in range(nSlinks):
        sum_low=0
        sum_mid=0
        sum_high=0
        sum_doub_low=0
        sum_doub_mid=0
        sum_doub_high=0
        list=[]
        for i in hml_single[:,j]:
            if i in lowS.values():
                sum_low+=1
        list.append(sum_low)
        for i in hml_single[:,j]:
            if i in medS.values():
                sum_mid+=1
        list.append(sum_mid)
        for i in hml_single[:,j]:
            if i in highS.values():
                sum_high+=1
        list.append(sum_high)
        for i in hml_double[:,j]:
            if i in lowD.values():
                sum_doub_low+=1
        list.append(sum_doub_low)
        for i in hml_double[:,j]:
            if i in medD.values():
                sum_doub_mid+=1
        list.append(sum_doub_mid)
        for i in hml_double[:,j]:
            if i in highD.values():
                sum_doub_high+=1
        list.append(sum_doub_high)

        a.append(list)
    b=np.array(a)
    c=b.T
    tablo=pd.DataFrame(c,index=["Single Low","Single Mid","Single High","Double Low","Double Mid","Double High"],
                          columns=["Slink"+str(i) for i in range(nSlinks)])
    tablo.index.name="lpgbt Rates"

    return tablo
print(create_dataframe())


# In[ ]:


sns.scatterplot(data=df_together, x="total rate", y="Lpgbt Rates", hue="link variable")
#sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.axvline(linewidth=1, color='r',x=totalRate / nSlinks)
'''
