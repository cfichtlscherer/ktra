#!/bin/bash
# Starting the detector response function simulations.

# steps, alpha, perc_noise, obj

# note to myself: Christopher you idiot learn bash loops!

cp -r block 20-0.105-0.01-3
cp -r block 20-0.110-0.01-3
cp -r block 20-0.115-0.01-3
cp -r block 20-0.120-0.01-3
cp -r block 20-0.125-0.01-3
cp -r block 20-0.130-0.01-3
cp -r block 20-0.135-0.01-3
cp -r block 20-0.140-0.01-3
cp -r block 20-0.145-0.01-3
cp -r block 20-0.150-0.01-3

cp -r block 20-0.105-0.02-3
cp -r block 20-0.110-0.02-3
cp -r block 20-0.115-0.02-3
cp -r block 20-0.120-0.02-3
cp -r block 20-0.125-0.02-3
cp -r block 20-0.130-0.02-3
cp -r block 20-0.135-0.02-3
cp -r block 20-0.140-0.02-3
cp -r block 20-0.145-0.02-3
cp -r block 20-0.150-0.02-3

cp -r block 20-0.105-0.03-3
cp -r block 20-0.110-0.03-3
cp -r block 20-0.115-0.03-3
cp -r block 20-0.120-0.03-3
cp -r block 20-0.125-0.03-3
cp -r block 20-0.130-0.03-3
cp -r block 20-0.135-0.03-3
cp -r block 20-0.140-0.03-3
cp -r block 20-0.145-0.03-3
cp -r block 20-0.150-0.03-3

cd 20-0.105-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.110-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.115-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.120-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.125-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.130-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.135-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.140-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.145-0.01-3 && python3 douglas_rachford_multi.py &\
cd 20-0.150-0.01-3 && python3 douglas_rachford_multi.py &\

cd 20-0.105-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.110-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.115-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.120-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.125-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.130-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.135-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.140-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.145-0.02-3 && python3 douglas_rachford_multi.py &\
cd 20-0.150-0.02-3 && python3 douglas_rachford_multi.py &\

cd 20-0.105-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.110-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.115-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.120-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.125-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.130-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.135-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.140-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.145-0.03-3 && python3 douglas_rachford_multi.py &\
cd 20-0.150-0.03-3 && python3 douglas_rachford_multi.py 

