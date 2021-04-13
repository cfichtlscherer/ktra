#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.1-0.01-2
cp -r block 20-0.15-0.01-2
cp -r block 20-0.2-0.01-2
cp -r block 20-0.25-0.01-2
cp -r block 20-0.3-0.01-2

cp -r block 20-0.1-0.02-2
cp -r block 20-0.15-0.02-2
cp -r block 20-0.2-0.02-2
cp -r block 20-0.25-0.02-2
cp -r block 20-0.3-0.02-2

cp -r block 20-0.1-0.03-3
cp -r block 20-0.15-0.03-3
cp -r block 20-0.2-0.03-3
cp -r block 20-0.25-0.03-3
cp -r block 20-0.3-0.03-3

cp -r block 20-0.1-0.04-3
cp -r block 20-0.15-0.04-3
cp -r block 20-0.2-0.04-3
cp -r block 20-0.25-0.04-3
cp -r block 20-0.3-0.04-3

cp -r block 20-0.1-0.05-3
cp -r block 20-0.15-0.05-3
cp -r block 20-0.2-0.05-3
cp -r block 20-0.25-0.05-3
cp -r block 20-0.3-0.05-3

cp -r block 20-0.1-0.01-3
cp -r block 20-0.15-0.01-3
cp -r block 20-0.2-0.01-3
cp -r block 20-0.25-0.01-3
cp -r block 20-0.3-0.01-3

cp -r block 20-0.1-0.02-3
cp -r block 20-0.15-0.02-3
cp -r block 20-0.2-0.02-3
cp -r block 20-0.25-0.02-3
cp -r block 20-0.3-0.02-3

cp -r block 20-0.1-0.03-3
cp -r block 20-0.15-0.03-3
cp -r block 20-0.2-0.03-3
cp -r block 20-0.25-0.03-3
cp -r block 20-0.3-0.03-3

cp -r block 20-0.1-0.04-3
cp -r block 20-0.15-0.04-3
cp -r block 20-0.2-0.04-3
cp -r block 20-0.25-0.04-3
cp -r block 20-0.3-0.04-3

cp -r block 20-0.1-0.05-3
cp -r block 20-0.15-0.05-3
cp -r block 20-0.2-0.05-3
cp -r block 20-0.25-0.05-3
cp -r block 20-0.3-0.05-3


cd 5-0.1-0.01-2     && python3 douglas_rachford_multi.py &\
cd 5-0.15-0.01-2  && python3 douglas_rachford_multi.py &\
cd 5-0.2-0.01-2  && python3 douglas_rachford_multi.py &\
cd 5-0.25-0.01-2  && python3 douglas_rachford_multi.py &\
cd 5-0.3-0.01-2  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.02-2  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.02-2  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.02-2  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.02-2  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.02-2  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.05-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.05-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.05-3  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.05-3 && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.05-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.04-3  && python3 douglas_rachford_multi.py &\
cd 20-0.1-0.05-3 && python3 douglas_rachford_multi.py &\
cd 20-0.15-0.05-3 && python3 douglas_rachford_multi.py &\
cd 20-0.2-0.05-3 && python3 douglas_rachford_multi.py &\
cd 20-0.25-0.05-3 && python3 douglas_rachford_multi.py &\
cd 20-0.3-0.05-3 && python3 douglas_rachford_multi.py 
