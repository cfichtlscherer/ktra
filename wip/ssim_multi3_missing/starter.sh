#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.115-0.02-3
cp -r block 20-0.115-0.03-3
cp -r block 20-0.130-0.01-3

cd 20-0.115-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.115-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.130-0.01-3 && python3 douglas_rachford_multi.py 

