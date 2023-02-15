from PG import master,Pyomo,Gekko
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--pyomo",default=False, action='store_true')
ap.add_argument("-g", "--gekko",default=False, action='store_true')

ap.add_argument("-m", "--m",help="lower bound")

ap.add_argument("-lb", "--lb",help="lower bound")
ap.add_argument("-ub", "--ub",help="upper bound")
args = vars(ap.parse_args())



mode='HtoL' #{'HtoL': high to low, 'LtoH': low to high, 'Ht': highest, 'Lt': lowest}

GekkoIter=2000

## Pyomo Running for different Slink rate limits
[single, double ] =master(mode, args["pyomo"], args["gekko"])

if args["pyomo"]:
    print("Start Pyomo!")
    i=float(args["m"])
    print(i)
    print(Pyomo(i, single, double))

## Gekko Running for different Slink rate limits

if args["gekko"]:
    print("Start Gekko!")
    i=float(args["m"])
    print(i)
    print(Gekko(i, single, double, GekkoIter))


'''
## Pyomo Running for different Slink rate limits	

if args["pyomo"]:
    print("Start Pyomo!")
    i=float(args["lb"])
    PyoDict={}
    while(i<=float(args["ub"])):
        print(i)
        PyoDict[i]=Pyomo(i)
        i+=.1
    print(PyoDict)
## Gekko Running for different Slink rate limits

if args["gekko"]:
    print("Start Gekko!")
    i=float(args["lb"])
    GekkoDict={}
    while(i<=float(args["ub"])):
        print(i)
        try:
            GekkoDict[i]=Gekko(i) 
            #print(std)
        except: continue
        i+=0.1
    print(GekkoDict)
'''
