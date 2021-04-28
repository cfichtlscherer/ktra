#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.005-0.01-3
cp -r block 20-0.010-0.01-3
cp -r block 20-0.015-0.01-3
cp -r block 20-0.020-0.01-3
cp -r block 20-0.025-0.01-3
cp -r block 20-0.030-0.01-3
cp -r block 20-0.035-0.01-3
cp -r block 20-0.040-0.01-3
cp -r block 20-0.045-0.01-3
cp -r block 20-0.050-0.01-3

cp -r block 20-0.005-0.02-3
cp -r block 20-0.010-0.02-3
cp -r block 20-0.015-0.02-3
cp -r block 20-0.020-0.02-3
cp -r block 20-0.025-0.02-3
cp -r block 20-0.030-0.02-3
cp -r block 20-0.035-0.02-3
cp -r block 20-0.040-0.02-3
cp -r block 20-0.045-0.02-3
cp -r block 20-0.050-0.02-3

cp -r block 20-0.005-0.03-3
cp -r block 20-0.010-0.03-3
cp -r block 20-0.015-0.03-3
cp -r block 20-0.020-0.03-3
cp -r block 20-0.025-0.03-3
cp -r block 20-0.030-0.03-3
cp -r block 20-0.035-0.03-3
cp -r block 20-0.040-0.03-3
cp -r block 20-0.045-0.03-3
cp -r block 20-0.050-0.03-3

cd 20-0.005-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.010-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.015-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.020-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.025-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.030-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.035-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.040-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.045-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.050-0.01-3  && python3 douglas_rachford_multi.py &\

cd 20-0.005-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.010-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.015-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.020-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.025-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.030-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.035-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.040-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.045-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.050-0.02-3  && python3 douglas_rachford_multi.py &\

cd 20-0.005-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.010-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.015-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.020-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.025-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.030-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.035-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.040-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.045-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.050-0.03-3  && python3 douglas_rachford_multi.py

