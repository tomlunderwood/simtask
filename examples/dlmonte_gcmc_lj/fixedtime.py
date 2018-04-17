import logging

import simtask.dlmonteinterface as interface
import simtask.measurement as measurement
import simtask.analysis as analysis
import simtask.task as task

# This script performs multiple DL_MONTE simulatins to calculate the mean and uncertainty in the energy and 
# number of molecules in the system reflected in the CONTROL, CONFIG and FIELD files in the current directory 
# - which correspond to a Lennard-Jones fluid above the critical temperature. The total number of simulations
# is constrained for convenience to take 30 seconds

# Make sure that the 'task' module is included the PYTHONPATH environment variable before executing this script,
# as well as Kevin Stratford's 'htk' module (which is imported by 'task')

# Set up the logger, which determines the nature of information output by the machinery in the 'task' package.
# The code below results in logging information being output to stdout

handler = logging.StreamHandler()

# Replacing 'logging.INFO' with 'logging.DEBUGGING' below results in more information being output by the
# logger. Using 'logging.WARNING' results in less information being output: only 'warnings'

measurement.logger.setLevel(logging.INFO)
measurement.logger.addHandler(handler)

# Set up the relevant TaskInterface object: which tells the low-level machinery in the 'task' package
# which code will be used to perform the simulations, and how to perform various tasks specific to that
# code, e.g. extracting the energy from output files created by the code.
# In this case we use DL_MONTE to perform our simulations; thus the TaskInterface object we will use
# is in fact a DLMonteInterface object (DLMonteInterface is a subclass of TaskInterface). The line
# below sets up a DL_MONTE-specific interface. Note that the interface must know the location of the 
# DL_MONTE executable - which is specified as the argument to the DLMonteInterface constructor.

interface = interface.DLMonteInterface("/Users/tlu20/DL_MONTE/dl_monte_v2_workspace/dlmonte2/bin/DLMONTE-PRL.X")

# Set up a list of 'observables' to track and analyse. Observables must be Observable objects, and the nature
# of Observable objects may vary between simulation codes. For DL_MONTE only observables corresponding to variables
# output periodically in YAMLDATA are currently supported. For a variable 'foo' specified in the YAMLDATA file
# the corresponding Observable object is returned by the command 'task.Observable( ("foo",) )'. Note the essential
# comma after "foo"! For a variable in YAMLDATA which is an array (e.g., 'nmol'), the observable corresponding to 
# the nth element in the array is returned by the command 'task.Observable( ("foo",n-1) )'. See below: 'energy_obs'
# corresponds to the 'energy' variable in YAMLDATA, and 'nmol_obs' corresponds to the 1st element in the 'nmol'
# array in YAMLDATA (which in fact is the number of molecules belonging to the 1st molecular species)

energy_obs = task.Observable( ("energy",) )
nmol_obs = task.Observable( ("nmol",0) )
observables = [ energy_obs, nmol_obs ]

# Set up the relevant Measurement object - which will actually perform the simulations and data analysis.
# Note that all the simulations and output files pertaining to analysis will be created in the directory
# 'fixedtime' (via the 'outputdir' argument). Furthermore, we specify that no more than 20 simulations will 
# be performed (via the 'axsims' argument), and that the maximum time we will allow over all simulations is 
# 30s (via the 'maxtime' argument).

m = measurement.Measurement(interface, observables, maxsims=20, maxtime=30, outputdir='fixedtime')

# Run the task

m.run()

# Once the task is complete in the 'fixedtime' directory there will be directories 'sim_1', 'sim_2', etc.
# containing the input and output files corresponding to each simulation. Note that 'sim_2' is a resumption
# of 'sim_1', 'sim_3' is a resumption of 'sim_2', etc. Furthermore the files 'energy_converge.dat' and
# 'nmol_0_converge.dat' contain, respectively, the mean and uncertainty in the energy and number of molecules
# after each simulation. Note that the uncertainty decreases as expected after each simulation

