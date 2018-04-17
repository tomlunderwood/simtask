"""Tests for 'analysis' module"""

import unittest
import numpy as np
import sys

import analysis


class AnalysisTestCase(unittest.TestCase):

    """Tests for 'analysis' module"""


    def test_block_averages(self):

        """Tests for 'block_averages' function"""

        data = ( -7.28820535515628000E+01,
                 -3.18863042402020000E+01,
                 2.19403616931461000E+01,
                 -6.40900178434064000E+01,
                 -5.95505218037951000E+01,
                 -7.39927727273677000E+00,
                 -2.02095006700651000E+01,
                 -9.67146259210056000E+01,
                 -4.88034291757944000E+01,
                 -1.34088503173081000E+01 )

        avgs_for_3 = ( -2.76093320328729000E+01,
                       -4.36799389733127000E+01,
                       -5.52425185889550000E+01 )

        avgs_rev_3 = ( -5.29756351380360000E+01,
                       -2.90530999155323000E+01,
                       -2.46786534634874000E+01 )

        np.testing.assert_almost_equal( analysis.block_averages(data,1), data, decimal=10 )
        np.testing.assert_almost_equal( analysis.block_averages(data,3), avgs_for_3, decimal=10 )
        np.testing.assert_almost_equal( analysis.block_averages(data,3,reverse=True), avgs_rev_3, decimal=10 )
        self.assertEqual( len(analysis.block_averages(data,11)), 0 )



    def test_autocorrelation(self):

        """Tests for 'autocorrelation' function"""

        sys.stdout.write("\nWARNING: No test coverage yet for 'autocorrelation' function!\n")




    def test_inefficiency(self):

        """Tests for 'inefficiency' function"""

        sys.stdout.write("\nWARNING: No test coverage yet for 'inefficiency' function!\n")




    def test_equilibration_test(self):

        """Tests for 'equilibration_test' function"""

        sys.stdout.write("\nWARNING: No test coverage yet for 'equilibration_test' function!\n")
        



if __name__ == '__main__':

    unittest.main()

