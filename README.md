# Replication Code for Policy Learning with Competing Agents

This repository contains the code for replicating the figures and simulations in Policy Learning for Competing Agents.

To reproduce Figure 1, 2, and 3, run paper-figure-1.ipynb, paper-figure-2.ipynb, and paper-figure-3.ipynb, respectively.

To create the environment for running the code/notebooks:
```
conda env create -n competing_agents --file environment.yml
```

To run the low-dimensional/toy experiments, use the following command.
```
./run_low_dim.sh
```
and reproduce the results from Table 1 using paper-table-1.ipynb.

To run high-dimensional experiments, use the following command. 
```
./run_high_dim.sh
```
and reproduce results from Figure 4 and Table 2 using paper-figure-4.ipynb and paper-table-2.ipynb, respectively.
