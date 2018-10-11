"""Tests for task base classes"""

import unittest

import task




class TaskTestCase(unittest.TestCase):

    """Tests for task base classes"""


    def test_observable(self):
        
        """Tests for Observable class"""

        obs1 = task.Observable( ("foo", 2, True) )
        obs2 = task.Observable( ("foo", 2, True) )
        obs3 = task.Observable( ("foot", 2, True) )
        obs4 = task.Observable( ("foo", 2, True, "bar") )

        # Test __eq__
        self.assertEqual(obs1, obs2)
        self.assertNotEqual(obs1, obs3)
        self.assertNotEqual(obs1, obs4)

        # Test that Observable objects can be used as keys in dictionaries (which
        # requires __eq__ and __hash__), i.e., Observable objects are hashable -
        # an exception is raised if they are not
        dict = { obs1 : 1, obs2 : 2, obs3 : 3, obs4 : 4 }




    def test_interface(self):

        """Tests for TaskInterface class"""

        interface = task.TaskInterface()
        
        self.assertRaises(NotImplementedError, interface.copy_input_files, "foo", "foo")
        self.assertRaises(NotImplementedError, interface.run_sim, "foo")
        self.assertRaises(NotImplementedError, interface.resume_sim, "foo", "foo")
        self.assertRaises(NotImplementedError, interface.extract_data, "foo", "foo")
        self.assertRaises(NotImplementedError, interface.amend_input_parameter, "foo", "foo","foo")
        


        
    def test_task(self):

        """Tests for Task class"""

        interface = task.TaskInterface()        
        t = task.Task(interface)

        self.assertRaises(NotImplementedError, t.run)




if __name__ == '__main__':

    unittest.main()
