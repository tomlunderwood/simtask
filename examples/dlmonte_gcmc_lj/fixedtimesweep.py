import logging

import simtask.dlmonteinterface as interface
import simtask.measurement as measurement
import simtask.analysis as analysis
import simtask.task as task

# See 'fixedtime.py' before considering this script

# This script performs multiple DL_MONTE simulations to calculate the mean and uncertainty in the energy and 
# number of molecules in the system reflected in the CONTROL, CONFIG and FIELD files in the current directory 
# - which correspond to a Lennard-Jones fluid - over a range of temperatures to obtain the energy vs. temperature
# and density vs. temperature for this system. The simulations used to calculate the mean and uncertainty at each 
# temperature are constrained for convenience to take 30 seconds. The 10 temperatures from 8000 to 17000 
# (inclusive) in gaps of 1000 are considered; thus this script should take about 300s = 5 mins to complete.

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

interface = interface.DLMonteInterface("/home/tom/Work/Code_workspace/DL_MONTE_2_dev/dlmonte2/bin/DLMONTE-PRL.X")

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

# Set up a Measurement object which will determine the nature of the simulations and data analysis at each
# temperature. We specify that no more than 20 simulations will be performed (via the 'axsims' argument), and 
# that the maximum time we will allow over all simulations at a given temperature is 30s (via the 'maxtime' 
# argument).

measurement_template = measurement.Measurement(interface, observables, maxsims=20, maxtime=30)

# Set up the list of temperatures to consider

temperatures = [8000,9000,10000,11000,12000,13000,14000,15000,16000,17000]

# Set up a MeasurementSweep object which - which will actually perform the simulations and data analysis.
# Note that all the simulations and output files pertaining to analysis will be created in the directory
# 'fixedtimesweep' (via the 'outputdir' argument). Note also that the control parameter 'param' is set to 
# "temperature" while the list of control parameters to explore 'paramvalues' is set to the the temperatures
# mentioned above. The DLMonteInterface, linked to the MeasurementSweep object via 'interface' in 
# 'measurement_template', thus knows to treat the temperature in the CONTROL file as a control parameter, 
# and explore the temperatures in 'temperatures' accordingly

sweep = measurement.MeasurementSweep(param="temperature", paramvalues=temperatures, 
                                     measurement_template=measurement_template, outputdir="fixedtimesweep")

# Run the task

sweep.run()


# Once the task is complete in the 'fixedtimesweep' directory there will be directories 'param_8000', 'param_9000',
# etc. containing the data pertaining to each temperature. The contents of each of these directories is the
# same as described in 'fixedtime.py'. However the salient results are contained in the files 'energy_sweep.dat'
# and 'nmol_0_sweep.dat', which contain plots of, respectively, the energy and its uncertainty vs. temperature,
# and the number of molecules in the system and its uncertainty (which is proportional to the density of the system)
# vs. temperature. Note that the 'task' module automatically detects whether the system has equilibrated, and
# also automatically determines a block size to use in block averaging. Thus it is possible that a simulation at one
# of the considered temperatures does not equilibrate within the allowed simulation time, or that it does equilibrate
# and that there is not enough post-equilibration data to obtain the mean or uncertainty. If the mean, say, energy
# cannot be calculated at a given temperature then that temperature is omitted from 'energy_sweep.dat'. If the mean
# can be calculated but the uncertainty cannot (note that the uncertainty requires more post-equilibration to
# calculate) then the uncertainty is quoted in 'energy_sweep.dat' as 'NaN'

