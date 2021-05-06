#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.055-0.01-3
cp -r block 20-0.060-0.01-3
cp -r block 20-0.065-0.01-3
cp -r block 20-0.070-0.01-3
cp -r block 20-0.075-0.01-3
cp -r block 20-0.080-0.01-3
cp -r block 20-0.085-0.01-3
cp -r block 20-0.090-0.01-3
cp -r block 20-0.095-0.01-3
cp -r block 20-0.100-0.01-3

cp -r block 20-0.055-0.02-3
cp -r block 20-0.060-0.02-3
cp -r block 20-0.065-0.02-3
cp -r block 20-0.070-0.02-3
cp -r block 20-0.075-0.02-3
cp -r block 20-0.080-0.02-3
cp -r block 20-0.085-0.02-3
cp -r block 20-0.090-0.02-3
cp -r block 20-0.095-0.02-3
cp -r block 20-0.100-0.02-3

cp -r block 20-0.055-0.03-3
cp -r block 20-0.060-0.03-3
cp -r block 20-0.065-0.03-3
cp -r block 20-0.070-0.03-3
cp -r block 20-0.075-0.03-3
cp -r block 20-0.080-0.03-3
cp -r block 20-0.085-0.03-3
cp -r block 20-0.090-0.03-3
cp -r block 20-0.095-0.03-3
cp -r block 20-0.100-0.03-3

cd 20-0.055-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.060-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.065-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.070-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.075-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.080-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.085-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.090-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.095-0.01-3  && python3 douglas_rachford_multi.py &\
cd 20-0.100-0.01-3  && python3 douglas_rachford_multi.py &\

cd 20-0.055-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.060-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.065-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.070-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.075-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.080-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.085-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.090-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.095-0.02-3  && python3 douglas_rachford_multi.py &\
cd 20-0.100-0.02-3  && python3 douglas_rachford_multi.py &\

cd 20-0.055-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.060-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.065-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.070-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.075-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.080-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.085-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.090-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.095-0.03-3  && python3 douglas_rachford_multi.py &\
cd 20-0.100-0.03-3  && python3 douglas_rachford_multi.py 

