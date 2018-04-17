""" Functions for analysing Monte Carlo time series """




import numpy as np
import logging

import scipy.special


logger = logging.getLogger(__name__)


# Values of the student's t-distribution corresponding to 95% confidence
# interval (two tailed), for t-distributions with various degrees of freedom. 
# TDIST[n] is defined as follows: the student's t-distribution with 'n'
# degrees of freedom will be between -TDIST[n] and TDIST[n] is 95%.
# The maximum 'n' tabulated here is 100. Note that the analogous value for
# 'n' in the infinite limit is 1.960.
TDIST = ( float('NaN'), 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
          2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
          2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060,
          2.056, 2.052, 2.048, 2.045, 2.042, 2.040, 2.037, 2.035, 2.032,
          2.030, 2.028, 2.026, 2.024, 2.023, 2.021, 2.020, 2.018, 2.017,
          2.015, 2.014, 2.013, 2.012, 2.011, 2.010, 2.009, 2.008, 2.007,
          2.006, 2.005, 2.004, 2.003, 2.002, 2.002, 2.001, 2.000, 2.000,
          1.999, 1.998, 1.998, 1.997, 1.997, 1.996, 1.995, 1.995, 1.994,
          1.994, 1.993, 1.993, 1.993, 1.992, 1.992, 1.991, 1.991, 1.990,
          1.990, 1.990, 1.989, 1.989, 1.989, 1.988, 1.988, 1.988, 1.987,
          1.987, 1.987, 1.986, 1.986, 1.986, 1.986, 1.985, 1.985, 1.985,
          1.984, 1.984, 1.984 )




def tdist(n):

    """Returns the value of the student's t-distribution corresponding to
    a 95% confidence interval

    Returns the value of the student's t-distribution corresponding to
    a 95% confidence interval (two-tailed) and `n` degrees of freedom.

    Parameters
    ----------
    n : int
        The number of degrees of freedom for the t-distribution

    Returns
    -------
    float
       The value corresponding to the two-tailed 95% confidence interval.
       For `n>100` the value corresponding to the normal distribution
       (which is the t-distribution in the limit of infinite degrees of
       freedom) is returned, namely 1.960. For `n<1` NaN is returned.

    Notes
    -----
    * For a given `n`, this function returns the two-tailed 95% confidence  
      interval for the student's t-distribution with 'n' degrees of freedom.
      This is the value :math:`x` for which the probability that the 
      t-distribution with `n` degrees of freedom is between :math:`-x` and
      :math:`x` is 95%
    * The values returned by this function are accurate to only 3 decimal
      places

    """

    if n > 100:

        return 1.960
 
    elif n <= 0:

        return float('NaN')

    else:

        return TDIST[n]




def block_averages(y, blocksize, reverse=False):

    """Extracts block averages from a time series
    
    Partitions the time series `y` into contiguous blocks, 
    calculates the average for each block, and returns these
    averages

    Parameters
    ----------
    y : array
        The time series under consideration
    blocksize : int
        The number of elements in `y` to include in a block
    reverse : bool
        If False (default) then blocks are constructed by iterating
        over elements in `y` starting from the first element in
        `y`. If True then blocks are constructed by iterating
        over elements in `y` starting from the last element in
        `y`, and proceeding to the first.
    
    Returns
    -------
    array
       An array containing the averages for all blocks

    Notes
    -----
    * The array returned by this function will be empty if
      `blocksize` is larger than the size of `y`    
    * Depending on the block size, if `reverse` is False then some
      elements at the end of `y` may not being included in any block.
      Conversely if `reverse` is True then some elements at the 
      beginning of `y` may not be included in any block.


    """

    logger.debug("Entered 'block_averages' function")
    logger.debug("len(y) = "+str(len(y)))
    logger.debug("y[0:9] = "+str(y[0:9]))
    logger.debug("blocksize = "+str(blocksize))
    logger.debug("reverse = "+str(reverse))

    # The number of blocks in the data
    nblocks = len(y) // blocksize

    logger.debug("nblocks = "+str(nblocks))

    if nblocks < 1:
        logger.debug("DEBUG WARNING: Found "+str(nblocks)+" blocks during block averaging")

    averages = np.zeros(nblocks)

    if reverse:

        for i in range(0,nblocks):
            logger.debug("block "+str(i)+": range = ["+str(len(y)-(i+1)*blocksize)+":"+str(len(y)-i*blocksize)+"]")
            averages[i] = np.mean( y[ (len(y)-(i+1)*blocksize) : (len(y)-i*blocksize) ] )
            logger.debug("block "+str(i)+": mean = "+str(averages[i]))

    else:

        for i in range(0,nblocks):
            logger.debug("block "+str(i)+": range = ["+str(i*blocksize)+":"+str((i+1)*blocksize)+"]")
            averages[i] = np.mean( y[ (i*blocksize) : ((i+1)*blocksize) ] )
            logger.debug("block "+str(i)+": mean = "+str(averages[i]))

    logger.debug("Exiting 'block_averages'")

    return averages




def autocorrelation(y):

    """Calculates the autocorrelation function of a time series

    Calculates the autocorrelation function of a time series. The
    definition of the autocorrelation function used here is given
    below

    Parameters
    ----------
    y : array
        The time series under consideration

    Returns
    -------
    array
        The autocorrelation function for `y`, as defined below, for 
        :math:`k=0,1,\dotsc,(N-1)`, where :math:`N` is the number
        of elements in `y`

    Notes
    -----
    * The autocorrelation function is defined here as
      :math:`\Psi_k=\frac{1}{N\text{Var}(y)}\sum_{i=1}^N(y_{i+k}-\bar{y})(y_i-\bar{y})`,
      where :math:`\bar{y}` is the mean in `y` and
      :math:`\text{Var}(y)=\sqrt{\frac{1}{N}\sum_{i=1}^N(y_i-\bar{y})^2}`
      is the variance in `y`
    * Note that the number of pairs of elements in `y` which are
      used to calculate :math:`\Psi_k` is :math:`(N-k)`. Hence
      the autocorrelation becomes more 'noisy' with increasing
      :math:`k`

    """

    logger.debug("Entered 'autocorrelation' function")
    logger.debug("len(y) = "+str(len(y)))
    logger.debug("y[0:9] = "+str(y[0:9]))

    mean =  np.mean(y)
    var = np.var(y)
    yshifted = y - mean

    logger.debug("mean = "+str(mean))
    logger.debug("var = "+str(var))
    logger.debug("yshifted[0:9] = "+str(yshifted[0:9]))

    # np.correlate(a,a,"full") returns c[k]=\sum_m a[m+k]*a[m] for
    # a real array 'a' of size 2*(N-1)+1 corresponding to the range of
    # k from -(N-1) to (N-1), where N is the size of 'a'. Element 0 in 
    # the returned array corresponds to k=-N+1, element 1 corresponds 
    # to k=-N+2, ...; element i corresponds to k=-N+1+i. Note that k=0
    # corresponds to element (N-1), and the highest element 
    # (2*(N-1)+1)-1 corresponds to k=(N-1) as expected.
    autocorr = np.correlate(yshifted, yshifted, "full") / (len(y)*var)

    # We want to return the autocorrelation function such that element i
    # corresponds to k=i, and ignoring elements corresponding to k<0.
    # This corresponds to an array comprised of elements (N-1) onwards
    # in autocorr
    logger.debug("len(autocorr) = "+str(len(autocorr)))
    logger.debug("returned array [0:9] = "+str(autocorr[(len(y)-1):(len(y)-1+9)]))
    logger.debug("Exiting 'autocorrelation'")
    return autocorr[(len(y)-1):]




def inefficiency(y):

    """Calculates the statistical inefficiency of data

    Calculates the statistical inefficiency of data. The
    definition of the statistical inefficiency used here is 
    given below

    Parameters
    ----------
    y : array
        The time series under consideration

    Returns
    -------
    float
        The statistical inefficiency of `y`, as defined below

    Notes
    -----
    * The statistical inefficiency is defined as :math:`s=1+2\sum_{k=1}^{\infty}\Phi_k`
      for a stationary infinite time series, where :math:`\Psi_k` is the autocorrelation 
      function (as defined in the function `autocorrelation` with :math:`N\to\infty`).
      It is a measure of the number of uncorrelated samples in the time series: an
      inefficiency of 1 signifies no correlations, while higher values signify increased
      correlations. Morevover the inefficiency is related to the autocorrelation time
      of the series. If we assume :math:`\Phi_k=\exp(-k/\tau)`, then it follows that
      :math:`s` and :math:`\tau` are related by :math:`\tau=-1/\ln[1-2/(s+1)]`.
    * Calculating :math:`s` directly via this formula (with the :math:`\infty` changed to
      :math:`(N-1)`, where :math:`N` is the size of the time series) will fail since 
      partial sums over :math:`k` do not converge. Here we use the 'initial positive 
      sequence estimator' of the sum - see the literature.
    * To calculate the statistical inefficiency of a time series as above, the time 
      series should be large enough that :math:`\Phi_k` can be estimated accurately
      enough

    """

    logger.debug("Entered 'inefficiency' function")
    logger.debug("len(y) = "+str(len(y)))
    logger.debug("y[0:9] = "+str(y[0:9]))
    
    logger.debug("Calculating autocorrelation function for statistical inefficiency...")
    autocorr = autocorrelation(y)
    logger.debug("len(autocorr) = "+str(len(autocorr)))
    logger.debug("autocorr[0:9] = "+str(autocorr[0:9]))

    # 's_sum' is the sum over 'k' mentioned above. Use the initial positive sequence 
    # estimator method to calculate this
    s_sum = 0.0
    for i in range(1,len(y)-1):

        logger.debug("i, s_sum, autocorr[i]+autocorr[i+1] = "+str(i)+" "+str(s_sum)+" "+str(autocorr[i]+autocorr[i+1]))

        if (autocorr[i]+autocorr[i+1])<0.0:
            break
        else:
            s_sum += autocorr[i]

    # The estimated statistical inefficiency
    s = 1.0+2.0*s_sum

    logger.debug("s = "+str(s))
    logger.debug("Exiting 'inefficiency'")

    return s




def equilibration_test(y, checktimes=[0.0,0.5], minslicesize=10, alpha=0.15, minslicesize_corrtime=6.0):

    """Checks whether a time series has equilibrated

    Checks whether a time series has equilibrated. This function considers slices of
    the data series in order to find a slice which is 'flat'. Here a slice is a window of
    the time series which can begin anywhere, but must end at the end of the time
    series. The slices the function will consider are specified in `checktimes`. If a slice
    is found to be flat, then the start time of the slice is a conservative estimate of the
    equilibration time for the time series.

    Whether or not a slice is flat is determined as follows. First, the mean and standard deviation
    of the slice are determined. The distance of first and last data points in the slice from
    the mean are then compared to the standard deviation. These distances are used to infer
    whether or not the slice is flat, the idea being that if there is, say, a significant increase
    throughout the slice, then the first data point will be far more than one standard deviation
    above the mean and the last data point will be far more than two standard deviations from
    the mean. A p-value is calculated corresponding to the locations of the first and last
    data points relative to the mean - where the null hypothesis is that the data points are
    drawn from a normal distribution with a mean and standard deviation corresponding to the whole
    slice. If this p-value is less than some threshold `alpha` then the positions of the first 
    and last data points are deemed not to correspond to the normal distribution - signifying that
    there is a trend. If the p-value is above the threshold then the positions of the first and
    last data points are deemed to correspond to the normal distribution - signifying that the
    slice is 'flat'.

    Parameters
    ----------
    y : array
        The time series under consideration
    checktimes : list
        A list of floats determining the slices of the time series to examine. Each
        element of the list corresponds to a slice, and is the fractional distance 
        along the time series at which the slice is to start. E.g. 0.5 corresponds 
        to a slice starting in the middle of the time series. The slices are examined
        in the order specified in the list, the function terminating if any slice 
        is found to be flat; thus the order of elements in the list should be in
        ascending order if one wishes to determine a more precise equilibration time
    minslicesize : int
        A slice with less than `minslicesize` elements will be deemed too small to 
        examine, and thus regarded as 'not flat'
    alpha : float
        Threshold for significance in p-value test. If the p-value is less than `alpha`
        then a slice is deemed to exhibit a trend which differs significantly
        from the null hypothesis, namely that the first and last data points in the
        slice are random variables corresponding to the normal distribution with a mean
        and variance deduced from the data in the slice, and accordingly the null 
        hypothesis is rejected (the slice is deemed not flat; the null hypothesis is
        that it is flat). Increasing `alpha` corresponds to the slice having to 'look 
        'more flat' for it to be deemed flat. `alpha` should be between 0.0 and 1.0,
        where 0.0 corresponds to no threshold for flatness and 1.0 corresponds to 
        only perfectly flat slices being deemed flat
    minslicesize_corrtime : float
        If a slice is found which is flat according to the p-value test, the slice 
        must also be sufficiently large compared to the autocorrelation time for the 
        slice before the slice is 'confirmed' to be flat: the number of data points 
        in the slice must be at least `minslicesize_corrtime` times the 
        autocorrelation time before the slice is confirmed to be flat

    Returns
    -------
    flatslice : boolean
        True if the function finds a flat slice, False if it does not
    slicepos : int
        If `flatslice` is True then `slicepos` is the location beyond which the
        time series was deemed flat, i.e. it is the time series index corresponding
        to the start of the slice which was deemed flat. This corresponds to one of 
        fractional locations specified in `checktimes` - *the first in that list 
        which yields a flat slice*. If `flatslice` is False then `slicepos` is the 
        location of the start of the last slice considered by the algorithm

    """

    logger.debug("Entered 'equilibration_test' function")
    logger.debug("len(y) = "+str(len(y)))
    logger.debug("y[0:9] = "+str(y[0:9]))

    flatslice = False

    for checktime in checktimes:

        logger.debug("Considering slice beginning at fraction "+str(checktime)+" of time series...")
        slicepos = int(checktime * len(y))
        slice = y[slicepos:]

        size = len(slice)

        logger.debug("  location of start of slice = "+str(slicepos))
        logger.debug("  size of slice = "+str(size))
        logger.debug("  first 10 data points in slice = "+str(slice[0:9]))

        if size < minslicesize:

            logger.debug("Slice is below size threshold. Assuming slice is not flat")
            break

        logger.debug("Slice is above size threshold")

        logger.debug("Calculating mean, variance and stdev. of slice...")
        mean = np.mean(slice)
        var = np.var(slice)
        stdev = np.sqrt(var)
        logger.debug("  mean, var, stdev = "+str(mean)+", "+str(var)+", "+str(stdev))

        first = slice[0]
        last = slice[-1]
        logger.debug("  first data point in slice = "+str(first))
        logger.debug("  last data point in slice = "+str(last))
        low = np.minimum(first,last)
        high = np.maximum(first,last)
        logger.debug("  lower data point = "+str(low))
        logger.debug("  higher data point = "+str(high))
        logger.debug("  lower point distance from mean in units of stdev = "+str((low-mean)/stdev))
        logger.debug("  higher point distance from mean in units of stdev = "+str((high-mean)/stdev))

        logger.debug("Starting p-value test...")
           
        # 'pvaluelow' is the two-tailed p-value associated with 'low': it is the probability
        # of sampling a point from a normal distribution with standard deviation 'stdev' which
        # is 'at least as far away from the mean' as 'low'. Similar applies for
        # 'pvaluehigh'
        pvaluelow = 0.5*(1.0+scipy.special.erf( -np.abs(mean-low)/(np.sqrt(2.0)*stdev) )) * 2
        pvaluehigh = 0.5*(1.0+scipy.special.erf( -np.abs(high-mean)/(np.sqrt(2.0)*stdev) )) * 2
        
        logger.debug("  p-values for lower and and higher points = "+str(pvaluelow)+", "+str(pvaluehigh))
        
        pvalue = pvaluelow * pvaluehigh
        logger.debug("  p-value for observing both points = "+str(pvalue))
        
        if pvalue < alpha:
             
            logger.debug("p-value less than alpha. Assuming slice is not flat")
            
        else:
            
            logger.debug("p-value greater than alpha. Assuming slice is flat")
            flatslice = True

        
        if flatslice:

            logger.debug("This slice seems flat. Cross-checking size of slice against correlation time...")

            logger.debug("Calculating autocorrelation time via statistical inefficiency...")
            s = inefficiency(slice)
            tau = -1.0/np.log(1.0-2.0/(s+1))
            logger.debug("  statistical inefficiency = "+str(s))
            logger.debug("  correlation time = "+str(tau))
            logger.debug("  slice size in units of correlation time = "+str(size/tau))

            if size > tau * minslicesize_corrtime:

                logger.debug("Slice is larger enough than correlation time threshold. Slice is confirmed flat.")
                logger.debug("Exiting 'equilibration_test'; returning: "+str(flatslice)+", "+str(slicepos))
                return flatslice, slicepos

            else:
            
                logger.debug("Slice is not larger enough than correlation time. Slice is no longer deemed flat")
                flatslice = False


        logger.debug("Slice is not deemed flat.")


    logger.debug("No flat slices found")
    logger.debug("Exiting 'equilibration_test'; returning: "+str(flatslice)+", "+str(slicepos))

    return flatslice, slicepos



