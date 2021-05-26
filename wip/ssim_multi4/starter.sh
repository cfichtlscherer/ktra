#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.155-0.01-3
cp -r block 20-0.160-0.01-3
cp -r block 20-0.165-0.01-3
cp -r block 20-0.170-0.01-3
cp -r block 20-0.175-0.01-3
cp -r block 20-0.180-0.01-3
cp -r block 20-0.185-0.01-3
cp -r block 20-0.190-0.01-3
cp -r block 20-0.195-0.01-3
cp -r block 20-0.200-0.01-3

cp -r block 20-0.155-0.02-3
cp -r block 20-0.160-0.02-3
cp -r block 20-0.165-0.02-3
cp -r block 20-0.170-0.02-3
cp -r block 20-0.175-0.02-3
cp -r block 20-0.180-0.02-3
cp -r block 20-0.185-0.02-3
cp -r block 20-0.190-0.02-3
cp -r block 20-0.195-0.02-3
cp -r block 20-0.200-0.02-3

cp -r block 20-0.155-0.03-3
cp -r block 20-0.160-0.03-3
cp -r block 20-0.165-0.03-3
cp -r block 20-0.170-0.03-3
cp -r block 20-0.175-0.03-3
cp -r block 20-0.180-0.03-3
cp -r block 20-0.185-0.03-3
cp -r block 20-0.190-0.03-3
cp -r block 20-0.195-0.03-3
cp -r block 20-0.200-0.03-3

cd 20-0.155-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.160-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.165-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.170-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.175-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.180-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.185-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.190-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.195-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.200-0.01-3 && python3 douglas_rachford_multi.py &\

cd 20-0.155-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.160-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.165-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.170-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.175-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.180-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.185-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.190-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.195-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.200-0.02-3 && python3 douglas_rachford_multi.py &\

cd 20-0.155-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.160-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.165-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.170-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.175-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.180-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.185-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.190-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.195-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.200-0.03-3 && python3 douglas_rachford_multi.py 

