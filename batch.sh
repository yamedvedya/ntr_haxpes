#!/bin/bash
#SBATCH --partition=all
#SBATCH --time=01:00:00                           # Maximum time requested
#SBATCH --nodes=2                                 # Number of nodes
#SBATCH --chdir   /home/matveyev/PycharmProjects/ntr_haxpes/data      # directory must already exist!
#SBATCH --job-name  mesh_solver
#SBATCH --output    mesh_solver-%j.out            # File to which STDOUT will be written
#SBATCH --error     mesh_solver-%j.err            # File to which STDERR will be written

mpiexec -n 2 /home/matveyev/PycharmProjects/ntr_haxpes/venv/bin/python /home/matveyev/PycharmProjects/ntr_haxpes/run_fit.py -s /home/matveyev/PycharmProjects/ntr_haxpes/data/solver.set