"""
Classes corresponding to a measurement of one or more observables at a given set of
conditions or thermodynamic ensemble
"""




import logging
import os
import time
import numpy as np
import copy

import scipy.stats
import scipy.special

import simtask.task as task
import simtask.analysis as analysis




logger = logging.getLogger(__name__)




class InsufficientDataError(Exception):

    """Exception corresponding to a simulation not providing enough data to
    proceed with data analysis

    Exception corresponding to a simulation not providing enough data to
    proceed with data analysis

    Attributes
    ----------
    message : str
        Human-readable message describing the exception

    """

    def __init__(self, message):

        self.message = message








class Measurement(task.Task):

    """
    Class corresponding to the task of 'measuring' one or more observables at a given set of
    conditions or thermodynamic ensemble

    Class corresponding to the task of 'measuring' one or more observables at a given set of
    conditions or thermodynamic ensemble. Here `run` performs simulations to calculate the 
    value and standard error one or more specified observables via block averaging. 

    The `run` function performs one or more simulations, gathering data from the output
    of the simulations for block averaging. Each simulation is performed in a separate
    directory, whose name is determined by `simdir_header`. If `simdir_header` were 'sim_',
    then the first simulation would be performed in 'sim_1', the second in 'sim_2', etc. 
    Each subsequent simulation corresponds to a continuation of the previous simulation, 
    and simulations are performed until one of the following three conditions is met:
    * A maximum number of simulations `maxsims` has been completed
    * A maximum time limit `maxtime` has been exceeded after any simulation
    * The standard errors for all observables are less than threshold values specified in 
      `precisions`

    The block size used for block averaging of each observable is determined automatically, using 
    `block_factor` for guidance. To elaborate, the autocorrelation time of each observable is 
    first determined automatically from the statistical inefficiency of the post-equilibration
    time series for the observable, and this autocorrelation time is then multiplied by
    `block_factor` to yield the block length for the observable. If `learn_equilibration_period` 
    is True then for each simulation the data for each observable is analysed to determine 
    whether or not the simulation has equilibrated or not. See `analysis.equilibration_test` 
    for details of this. If equilibration is deemed to have been reached, then block averaging 
    is performed using the post-equilibration data. If `learn_equilibration_period` is False 
    then the simulation is assumed to be equilibrated from the outset, i.e., all data is 
    regarded as post-equilibration.

    `run` creates a number of output files. Firstly, after each simulation the block averages 
    obtained for each observable 'obs' under consideration are output to a file 
    'obs_blockavgs.dat', which is located in the directory corresponding to that simulation. 
    E.g., the block averages corresponding to the observable 'energy' (if it were under 
    consideration) obtained from simulation 2 would be found in the file
    'sim_2/energy_blockavgs.dat'. *Note that the block averages for a given simulation in
    fact use post-equilibration data from all simulations up to and including that simulation*.

    Moreover, after each simulation the mean and standard error obtained from block averaging 
    for observable 'obs' is output to the file 'obs_convergence.dat' in the directory in 
    which `run` is invoked. Specifically, each line of the file contains the simulation number,
    followed by the mean and then standard error obtained from block averaging using 
    post-equilibration data for the observable *obtained from all simulations up to and 
    including that simulation*. Thus 'obs_convergence.dat' contains details of the convergence 
    of the mean and standard error for the variable 'obs' throughout the task. Note however 
    that output to the file is only made if both the mean and standard error are calculable 
    via block averaging, i.e. if there there at least a few blocks of post-equilibration data.


    Attributes - for controlling the nature of the task
    ---------------------------------------------------
    interface : task.TaskInterface
        TaskInterface object corresponding to the molecular simulation code to
        be used
    observables : tuple
        A set of observables (task.Observable objects) to be measured
    maxsims : int
        The maximum number of simulations to perform (default = 10)
    maxtime : int
        The maximum time this function is to run for, in seconds 
        (default = 604800, which corresponds to one week)
    precisions : dict
        A dictionary containing threshold precisions (floats) for 
        observables: no more simulations will be performed if the 
        standard errors in all observables corresponding to keys in
        `precisions` are below the values associated with the keys
        E.g., { task.Observable('energy') : 1.0E-6, task.Observable('volume') : 1.0E-4}
        corresponds to a precision of 1.0E-6 and 1.0E-4 for the energy
        and volume respectively. (default = {}, i.e., an empty 
        dictionary)
    learn_equilibration_period : boolean
        Flag determining whether an equilibration period is to be 
        deduced from the data. If False then the system is tacitly 
        assumed to be equilibrated from the outset (default = False)
    block_factor : float
        The block length to use - in units of the autocorrelation time for
        the observable under consideration - where the autocorrelation
        function is assumed to have the form :math:`e^{-t/\tau}`, where
        :math:`\tau` is the autocorrelation time, and here the unit of
        time is the time between observations in a time series obtained
        from `extract_data`. For a `block_factor` of :math:`n` the 
        correlation between observations in adjacent blocks is
        :math:`e^{-n}` or less. (default = 10)
    simdir_header : str
        The header for the names of the directories for the simulations to
        be performed as part of this task. E.g. if `simdir_header` were 'sim_',
        then the 1st simulation would be performed in 'sim_1', the second in
        'sim_2', etc. (default = 'sim_')
    inputdir : str
        The name of the directory containing the input files for the first
        simulation. (default = os.curdir, i.e., the current directory)
    outputdir : str
        The name of the directory where output files and simulation directories
        created by this task will be created. 
        (default = os.curdir, i.e., the current directory)
    equilibrate_only : bool
        If True then the task stop when the system has been deemed to 
        be equilibrated - without performing block averages. (default = False)


    Attributes for controlling the nature of the equilibration test (if in use)
    ---------------------------------------------------------------------------
    equiltest_checktimes : list
        (see `analysis.equilibration_test`) (default = [0.0,0.5])
    equiltest_minslice : int
        (see `analysis.equilibration_test`) (default = 10)
    equiltest_alpha : float
        (see `analysis.equilibration_test`) (default = 0.15)
    equiltest_corrtime : float
        (see `analysis.equilibration_test`) (default = 6.0)


    Attributes - 'read only'
    ------------------------
    data : dict
        'data[obs]' is the time series for observable 'obs' obtained from all
        simulations
    blockavgs : dict
        'blockavgs[obs]' is an array containing the block averages over all
        simulations for observable 'obs'
    blocksize : dict
        'blocksize[obs]' is the number of data points to use in a block for 
        block averaging - for observable 'obs'
    mean : dict
        'mean[obs]' is the mean value for observable 'obs' obtained from block
        averaging
    stderr : dict
        'stderr[obs]' is the standard error of the mean for observable 'obs' 
        obtained from block averaging
    equilibrated : dict
        'equilibrated[obs]' is a boolean which is True if the time series for
        'obs' has equilibrated and False if not
    equiltime : dict
        'equiltime[obs]' is the equilibration time for observable 'obs'
    simdirs : list
        List of directories in which simulations were completed

    """




    def __init__(self, task_interface, 
                 observables, 
                 maxsims=10, maxtime=604800, precisions={}, 
                 learn_equilibration_period=False, block_factor=10, 
                 simdir_header="sim_", inputdir=os.curdir, outputdir=os.curdir,
                 equilibrate_only=False,
                 equiltest_checktimes=[0.0,0.5], equiltest_minslice=10,
                 equiltest_alpha=0.15, equiltest_corrtime=6.0
                ):

        # Control variables
        self.interface = task_interface
        self.observables = observables
        self.maxsims = maxsims
        self.maxtime = maxtime
        self.precisions = precisions
        self.learn_equilibration_period = learn_equilibration_period
        self.block_factor = block_factor
        self.simdir_header = simdir_header
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.equilibrate_only = equilibrate_only
        self.equiltest_checktimes = equiltest_checktimes
        self.equiltest_minslice = equiltest_minslice
        self.equiltest_alpha = equiltest_alpha
        self.equiltest_corrtime = equiltest_corrtime

        # 'Read-only' variables 
        self.data = {}
        self.blockavgs = {}
        self.blocksize = {}
        self.mean = {}
        self.stderr = {}
        self.equilibrated = {}
        self.equiltime = {}
        self.simdirs = []




    def __str__(self):

        """Return a readable string representation of a Measurement object

        Return a string representation of Measurement object. Note that
        this does not output the `data` attribute.
        """
        
        string = "Class : "+self.__class__.__name__+"\n"
        string += "Control attributes:\n"
        string += "  interface : "+str(self.interface)+"\n"
        string += "  observables : "+", ".join("%s" % str(obs) for obs in self.observables)+"\n"
        string += "  maxsims : "+str(self.maxsims)+"\n"
        string += "  maxtime : "+str(self.maxtime)+"\n"
        string += "  precisions :\n"+"".join("    %s : %s\n" % (obs,self.precisions[obs]) for obs in self.precisions)
        string += "  learn_equilibration_period : "+str(self.learn_equilibration_period)+"\n"
        string += "  block_factor : "+str(self.block_factor)+"\n"
        string += "  simdir_header : "+str(self.simdir_header)+"\n"
        string += "  inputdir : "+str(self.inputdir)+"\n"
        string += "  outputdir : "+str(self.outputdir)+"\n"
        string += "  equilibrate_only : "+str(self.equilibrate_only)+"\n"
        string += "  equiltest_checktimes : "+str(self.equiltest_checktimes)+"\n"
        string += "  equiltest_minslice : "+str(self.equiltest_minslice)+"\n"
        string += "  equiltest_alpha : "+str(self.equiltest_alpha)+"\n"
        string += "  equiltest_corrtime : "+str(self.equiltest_corrtime)+"\n"
        string += "Data attributes:\n"
        string += "  blockavgs :\n"+"".join("    %s : %s\n" % (obs,self.blockavgs[obs]) for obs in self.blockavgs)
        string += "  blocksize :\n"+"".join("    %s : %s\n" % (obs,self.blocksize[obs]) for obs in self.blocksize)
        string += "  mean :\n"+"".join("    %s : %s\n" % (obs,self.mean[obs]) for obs in self.mean)
        string += "  stderr :\n"+"".join("    %s : %s\n" % (obs,self.stderr[obs]) for obs in self.stderr)
        string += "  equilibrated :\n"+"".join("    %s : %s\n" % (obs,self.equilibrated[obs]) for obs in self.equilibrated)
        string += "  equiltime :\n"+"".join("    %s : %s\n" % (obs,self.equiltime[obs]) for obs in self.equiltime)
        string += "Other attributes:\n"
        string += "  simdirs : "+", ".join("%s" % dir for dir in self.simdirs)+"\n"

        return string




    def run(self):

        """Perform the measurement task

        Perform the measurement task. See the class-level documentation for 
        details.

        """

        logger.info("")
        logger.info("Beginning measurement task...")
        logger.info("")

        logger.debug("Snapshot of this task:\n")
        logger.debug(self)
        logger.debug("")

        # Start time in seconds
        start_time = time.time()


        # Delete old task output files if they exist
        for obs in self.observables:

            # More pythonic to use try statement to remove file and catch OSError
            # if the file doesn't exist, then pass

            convergefilename = str(self.outputdir)+"/"+str(obs)+"_converge.dat"

            if os.path.exists(convergefilename):

                logger.warning("WARNING: Deleting file '"+str(convergefilename)+"'")
                os.remove(convergefilename)


        logger.info("")
        logger.info("Beginning simulations...")

        for i in range(0,self.maxsims):
            
            simno = i+1

            logger.info("")
            logger.info("")
            logger.info("On simulation "+str(simno)+" of max. "+str(self.maxsims)+"...")
            logger.info("")

            simdir = self.outputdir+"/"+self.simdir_header+str(simno)
            logger.info("Setting up directory '"+simdir+"' for simulation...")
            if not os.path.exists(self.outputdir):
                logger.debug("Creating directory '"+self.outputdir+"'...")
                os.mkdir(self.outputdir)
            logger.debug("Creating directory '"+simdir+"'...")
            os.mkdir(simdir)

            if simno==1:

                # We must start a new simulation...

                logger.debug("Copying input files to '"+simdir+"'...")
                self.interface.copy_input_files(self.inputdir, simdir)

                logger.info("Running simulation in '"+str(simdir)+"'...")
                self.interface.run_sim(simdir)
                logger.info("Simulation complete")

                self.simdirs.append(simdir)

            else:

                # We are resuming the previous simulation...

                logger.info("Running simulation in '"+str(simdir)+"'...")
                prevsimdir = self.outputdir+"/"+self.simdir_header+str(simno-1)
                logger.debug("(Resuming simulation from '"+str(prevsimdir)+"')")
                self.interface.resume_sim(prevsimdir, simdir)
                logger.info("Simulation complete")
            

            # Extract and analyse the data for each observable...
            for obs in self.observables:

                logger.info("")
                logger.info("Extracting data for observable '"+str(obs)+"'...")

                try:
                
                    simdata = self.interface.extract_data(obs, simdir)

                except:

                    logger.error("ERROR: Problem extracting data for observable '"+str(obs)+"' from '"+str(simdir)+"'")
                    raise
                
                
                # Check that there actually is data before proceeding with the analysis

                if len(simdata)==0:

                    logger.error("ERROR: No data was found for observable '"+str(obs)+"' in '"+str(simdir))
                    raise InsufficientDataError("No data found for observable '"+str(obs)+"' in '"+str(simdir))


                logger.debug("Info regarding data for '"+str(obs)+"' extracted from last simulation:")
                logger.debug("  number of data points = "+str(len(simdata)))
                logger.debug("  up to first 10 data points = "+str(simdata[:10]))
                logger.debug("  last data point = "+str(simdata[len(simdata)-1]))

                logger.debug("Appending to cummulative data for this observable...")

                # Need to initialise self.data[obs] as an empty list at the first simulation
                if simno==1:

                    self.data[obs] = []

                self.data[obs] = np.append(self.data[obs], simdata)

                logger.debug("Info regarding all extracted data for '"+str(obs)+"':")
                logger.debug("  number of data points = "+str(len(self.data[obs])))
                logger.debug("  up to first 10 data points = "+str(self.data[obs][:10]))
                logger.debug("  last data point = "+str(self.data[obs][len(self.data[obs])-1]))

                logger.info("Extraction complete")


                logger.info("Analysing time series for observable '"+str(obs)+"'...")


                if self.learn_equilibration_period:

                    logger.debug("Proceeding to equilibration test...")
                    
                    self.equilibrated[obs], self.equiltime[obs] = analysis.equilibration_test(self.data[obs])
                    logger.debug("Results of equilibration test:")
                    logger.debug("  equilibrated = "+str(self.equilibrated[obs]))
                    if self.equilibrated[obs]:

                        logger.debug("  conservative equilibration time = "+str(self.equiltime[obs]))
                    
                else:

                    logger.info("Bypassing equilibration test: equilibration assumed from outset")

                    self.equilibrated[obs] = True
                    self.equiltime[obs] = 0

                    
                if self.equilibrated[obs]:

                    logger.info("System is deemed equilibrated")
                    logger.info("  equilibration time = "+str(self.equiltime[obs]))

                    if self.equilibrate_only:

                        logger.info("Completed measurement task")
                        return 


                    logger.info("Deducing blocks to use for block averaging...")

                    logger.debug("Calculating autocorrelation time for observable '"+str(obs)+"'...")

                    inefficiency = analysis.inefficiency(self.data[obs][self.equiltime[obs]:])

                    # tau is the correlation time (assuming the corelation function is
                    # of the form exp(-k/tau))
                    tau = -1.0/np.log(1.0-2.0/(inefficiency+1))

                    self.blocksize[obs] = int(self.block_factor*tau)+1

                    logger.info("Results of post-equilibration time series analysis for observable '"+str(obs)+"':")
                    logger.info("  number of data points = "+str(len(self.data[obs])))
                    logger.info("  statistical inefficiency = "+str(inefficiency))
                    logger.info("  correlation time = "+str(tau))
                    logger.info("  calculated block size = "+str(self.blocksize[obs]))
                               
                    logger.info("Calculating block averages...")

                    self.blockavgs[obs] = analysis.block_averages(self.data[obs][self.equiltime[obs]:], self.blocksize[obs])
                    nblocks = len(self.blockavgs[obs])

                    logger.info("  number of blocks found = "+str(nblocks))   

                    # Output block averages to 'obs_blockavgs.dat'
                    blockfilename = str(simdir)+"/"+str(obs)+"_blockavgs.dat"
                    blockfile = open(blockfilename,'a')
                    logger.debug("Outputting block averages to '"+str(blockfilename)+"'...")
                    for i in range(0,nblocks):
                        blockfile.write(str(self.blockavgs[obs][i])+"\n")
                    blockfile.close()

                    
                    logger.info("Analysing block averages for observable '"+str(obs)+"' (for mean and standard error)...")

                    # We can only calculate the mean if we have at least one block
                    if len(self.blockavgs[obs]) >= 1:

                        logger.debug("Calculating mean...")

                        s = analysis.inefficiency(self.blockavgs[obs])
                        self.mean[obs] = np.mean(self.blockavgs[obs])

                        # We can only calculate the standard error if we have at least two blocks
                        if len(self.blockavgs[obs]) >= 2:

                            logger.debug("Calculating standard error...")

                            # EXPLAIN WHERE THIS VLAUE COMES FROM
                            self.stderr[obs] = np.sqrt( analysis.tdist(len(self.blockavgs[obs])) * 
                                          np.var(self.blockavgs[obs])*s/len(self.blockavgs[obs]) )

                            # Output mean and standard error to file
                            convergefilename = str(self.outputdir)+"/"+str(obs)+"_converge.dat"
                            logger.debug("Outputting mean and standard error to '"+str(convergefilename)+"'...")
                            convergefile = open(convergefilename, "a")
                            convergefile.write(str(simno)+" "+str(self.mean[obs])+" "+str(self.stderr[obs])+"\n")
                            convergefile.close()

                        else:

                            logger.info("Cannot calculate standard error: we need >= 2 post-equilibration blocks")

                            self.stderr[obs] = float('NaN')

                        logger.info("Results of analysis of block averages:")
                        logger.info("  statistical inefficiency of block averages = "+str(s))
                        logger.info("  mean from block averages = "+str(self.mean[obs]))
                        logger.info("  std. error from block averages = "+str(self.stderr[obs]))
                        

                    else:

                        logger.info("Cannot calculate mean: we need >= 1 post-equlibration block")

                else:
                
                    logger.info("Bypassing mean and standard error calculation: system is assumed not equilibrated for this observable")

            logger.info("Analysis complete")

            logger.debug("")
            logger.debug("Snapshot of this task:\n")
            logger.debug(self)
            logger.debug("")

            logger.info("")
            logger.info("Checking end-of-task criteria...")
            
            # Cycle over all keys in 'precisions' and check if the analogous 'stderror' 
            # is less than it

            logger.debug("Cycling over observables in 'precisions'...")

            reached_precisions = {}

            for obs in self.precisions:

                logger.debug("Considering observable '"+str(obs)+"'")

                # The keys in self.precisions may not actually correspond to any observables in
                # self.stderr. Assuming the user has not made an error in setting self.precisions
                # (e.g. by putting 'enerhy' as the key for the observable 'energy'), a KeyError
                # exception would occur below only if self.stderr[obs] has not been set, i.e., 
                # there have not been enough blocks yet to calculate self.stderr[obs]
                try:

                    obs_stderr = self.stderr[obs]

                except KeyError:
                    
                    logger.debug("A standard error value for this observable has not been found")

                else:

                    logger.debug("A standard error value for this observable has been found: "+str(self.stderr[obs]))
                    
                    logger.debug("Checking value against threshold...")
                    
                    if np.abs(self.stderr[obs]) < self.precisions[obs]:

                        reached_precisions[obs] = True
                        logger.info("Threshold precision for std. error of observable '"+str(obs)+"' reached")

                    else:

                        reached_precisions[obs] = False
                        logger.debug("Threshold precision for std. error of observable '"+str(obs)+"' not yet reached")


            # Check that if the dictionary 'reached_precisions' has any keys,
            # then are they all True. If so then all observables tracked for
            # precision have been calculated to the desired precision
            logger.debug("Dictionary holding booleans for precision threshold criteria:")
            logger.debug("  reached_precisions = "+str(reached_precisions))

            if len(reached_precisions) > 0  and  all( v==True for v in reached_precisions.values() ):

                logger.info("Precision threshold reached for all observables")
                break

            else:

                logger.debug("Precision threshold has not been reached for all observables, or threshold is not in play")


            elapsed_time = time.time() - start_time

            if elapsed_time > self.maxtime:

                logger.info("Task time threshold of "+str(self.maxtime)+"s exceeded:")
                logger.info("  elapsed time = "+str(elapsed_time)+"s")
                break

            
            if(simno==self.maxsims):
                
                logger.info("Reached maximum number of simulations")

            else:

                logger.info("End-of-task criteria not yet reached. Proceeding to next simulation...")


        logger.info("")
        logger.info("")
        logger.info("Completed measurement task")

        






class MeasurementSweep(task.Task):

    """
    Class corresponding to the task of performing multiple similar 'measurements' of
    one or more observables at various values of a control parameter

    Class corresponding to the task of performing multiple similar 'measurements' of
    one or more observables at various values of a control parameter

    Attributes - for controlling the nature of the task
    ---------------------------------------------------
    param : str
        The name of the 'control parameter' which will be varied between
        measurements
    paramvalues : list
        The values of the control parameter to be explored
    measurement_template : Measurement
        A Measurement object whose control attributes will be used as a template
        for the measurement at each control parameter listed in `paramvalues` 
    paramdir_header : str
        The header for the name of the directories to contain the simulations
        for each control parameter. (default = 'param_')
    outputdir : str
        The name of the directory where output files and simulation directories
        created by this task will be created. 
        (default = os.curdir, i.e., the current directory)

    Attributes - 'read only'
    ------------------------
    measurements : list
        A list of Measurement objects corresponding to each control parameter
        in `paramvalues`

    """




    def __init__(self, param, paramvalues, measurement_template, 
                 paramdir_header="param_", outputdir=os.curdir):

        self.param = param
        self.paramvalues = paramvalues
        self.measurement_template = measurement_template
        self.paramdir_header = paramdir_header
        self.outputdir = outputdir
        
        self.measurements = []




    def __str__(self):
        
        """Return a readable string representation of a MeasurementSweep object

        """

        string = "Class : "+self.__class__.__name__+"\n"
        string += "Simple control attributes:\n"
        string += "  param : "+str(self.param)+"\n"
        string += "  paramvalues : "+str(self.paramvalues)+"\n"
        string += "  paramdir_header : "+str(self.paramdir_header)+"\n"
        string += "START OF TEMPLATE MEASUREMENT INFO\n"+str(self.measurement_template)+"END OF TEMPLATE MEASUREMENT INFO\n"
        string += "".join( "START OF MEASUREMENT %s INFO (for paramval = %s)\n%sEND OF MEASUREMENT %s INFO\n" 
                           % (str(i), str(self.paramvalues[i]), str(self.measurements[i]), str(i)) 
                           for i in range(0,len(self.measurements)) 
                          )

        return string




    def run(self):

        """Perform the task

        Perform the task. See the class-level documentation for 
        details.

        """

        # TO DO: IMPROVE ERROR HANDLING

        logger.info("")
        logger.info("Beginning measurement sweep task...")
        logger.info("")
        
        logger.debug("Snapshot of this task:\n")
        logger.debug(self)
        logger.debug("")


        # Delete old task output files if they exist
        for obs in self.measurement_template.observables:

            # More pythonic to use try statement to remove file and catch OSError
            # if the file doesn't exist, then pass
            filename = self.outputdir+"/"+str(obs)+"_sweep.dat"
            if os.path.exists(filename):
                logger.warning("WARNING: Deleting file '"+str(filename)+"'")
                os.remove(filename)


        for val in self.paramvalues:
            
            logger.info("")
            logger.info("Beginning measurement for control parameter value "+str(val)+"...")
            logger.info("")

            paramdir = self.outputdir+"/"+self.paramdir_header+str(val)
            logger.info("Setting up directory '"+paramdir+"' for control parameter value "+str(val)+"...")
            if not os.path.exists(self.outputdir):
                logger.debug("Creating directory '"+self.outputdir+"'...")
                os.mkdir(self.outputdir)
            logger.debug("Creating directory '"+paramdir+"'...")
            os.mkdir(paramdir)

            # ERROR HANDLING FOR BAD USER INPUT, E.G., NON-EXISTENT CONTROL PARAMETERS?

            logger.debug("Copying input files from '"+self.measurement_template.inputdir+"' to '"+paramdir+"'...")
            self.measurement_template.interface.copy_input_files(self.measurement_template.inputdir, paramdir)

            logger.debug("Amending control parameter in "+str(paramdir)+" to correspond to value "+str(val)+"...")
            self.measurement_template.interface.amend_input_parameter(paramdir, self.param, val)
            
            logger.info("Running measurement for control parameter "+str(val)+"...")

            logger.debug("Creating new Measurement object from template to hold results...")
            measurement = copy.deepcopy(self.measurement_template)
            logger.debug("Amending new Measurement object's input directory to be '"+str(paramdir)+"'...")
            measurement.inputdir = paramdir
            logger.debug("Amending new Measurement object's output directory to be '"+str(paramdir)+"'...")
            measurement.outputdir = paramdir
            logger.debug("Attributes of new Measurement object for this control parameter:")
            logger.debug(str(measurement))
            logger.debug("")

            try:

                logger.debug("Calling run() for Measurement object...")
                measurement.run()

                logger.info("")                
                logger.info("")
                logger.info("Completed measurement for control parameter "+str(val))
                logger.info("")
                logger.info("")

            except InsufficientDataError:

                logger.error("Insufficient data for analysis from simulation for control parameter "+str(val))

            else:

                logger.info("Extracting mean and standard error for all observables for control parameter "+str(val)+"...")

                for obs in measurement.observables:

                    logger.info("")
                    logger.debug("Considering observable "+str(obs)+"...")
                    
                    # Only output if there is equilibration w.r.t. this observable, and we have
                    # obtained at least 1 block average
                    if measurement.equilibrated[obs] and len(measurement.blockavgs[obs]) >= 1:
                        
                        logger.info("Observable '"+str(obs)+"' was deemed equilibrated during measurement")
                        logger.info("Final result for observable:")
                        logger.info("  mean = "+str(measurement.mean[obs]))
                        logger.info("  standard error = "+str(measurement.stderr[obs]))

                        filename = self.outputdir+"/"+str(obs)+"_sweep.dat"
                        logger.debug("Outputting result to file '"+filename+"'...")
                        file = open(filename, 'a')
                        file.write(str(val)+" "+str(measurement.mean[obs])+" "+str(measurement.stderr[obs])+"\n")
                        file.close()

                    else:

                        logger.info("Observable '"+str(obs)
                                    +"' was not deemed equilibrated during measurement: bypassing extraction...")

            finally:

                # Update the list of measurements regardless of whether there was an InsufficientDataError
                # or not

                logger.debug("Appending list of completed measurements...")
                self.measurements.append(measurement)

                logger.info("")
                logger.info("Completed extraction of mean and standard error for control parameter "+str(val))
                logger.info("")

                logger.debug("Snapshot of this task:\n")
                logger.debug(self)
                logger.debug("")

        logger.info("")
        logger.info("Measurement sweep task complete")
        logger.info("")



