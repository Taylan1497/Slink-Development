#! /bin/bash

#:<<'Loop'

for i in $(seq 1.0 0.1 2.0)
do
    eval "python call_PG.py -p -m $i"
done

Loop

:<<'Array'
filearr=(1.8);


for file02 in ${filearr[@]}
do
	eval "python call_PG.py -p -m $file02"

done
Array


#eval "python call_PG.py -p -lb 1.5 -ub 1.5"
