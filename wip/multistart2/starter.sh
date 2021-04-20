#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.05-0.01-1
cp -r block 20-0.1-0.01-1
cp -r block 20-0.15-0.01-1
cp -r block 20-0.05-0.01-2
cp -r block 20-0.1-0.01-2
cp -r block 20-0.15-0.01-2
cp -r block 20-0.05-0.01-3
cp -r block 20-0.1-0.01-3
cp -r block 20-0.15-0.01-3

cp -r block 20-0.05-0.02-1
cp -r block 20-0.1-0.02-1
cp -r block 20-0.15-0.02-1
cp -r block 20-0.05-0.02-2
cp -r block 20-0.1-0.02-2
cp -r block 20-0.15-0.02-2
cp -r block 20-0.05-0.02-3
cp -r block 20-0.1-0.02-3
cp -r block 20-0.15-0.02-3

cp -r block 20-0.05-0.03-1
cp -r block 20-0.1-0.03-1
cp -r block 20-0.15-0.03-1
cp -r block 20-0.05-0.03-2
cp -r block 20-0.1-0.03-2
cp -r block 20-0.15-0.03-2
cp -r block 20-0.05-0.03-3
cp -r block 20-0.1-0.03-3
cp -r block 20-0.15-0.03-3


cd 20-0.05-0.01-1     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.01-1     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.01-1  && python3 douglas_rachford_multi.py &\
cd 20-0.05-0.01-2     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.01-2     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.01-2  && python3 douglas_rachford_multi.py &\
cd 20-0.05-0.01-3     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.01-3     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.01-3  && python3 douglas_rachford_multi.py &\

cd 20-0.05-0.02-1     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.02-1     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.02-1  && python3 douglas_rachford_multi.py &\
cd 20-0.05-0.02-2     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.02-2     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.02-2  && python3 douglas_rachford_multi.py &\
cd 20-0.05-0.02-3     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.02-3     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.02-3  && python3 douglas_rachford_multi.py &\

cd 20-0.05-0.03-1     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.03-1     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.03-1  && python3 douglas_rachford_multi.py &\
cd 20-0.05-0.03-2     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.03-2     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.03-2  && python3 douglas_rachford_multi.py &\
cd 20-0.05-0.03-3     && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.03-3     && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.03-3  && python3 douglas_rachford_multi.py &\

