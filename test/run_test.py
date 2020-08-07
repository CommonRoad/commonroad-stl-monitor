import unittest as unittest
import os

if __name__ == "__main__":
    for x in os.walk(os.getcwd()):
        if '__' not in x[0] and '.' not in x[0]:
            print(x[0])
            all_tests = unittest.TestLoader().discover(x[0], pattern='test_*.py')
            unittest.TextTestRunner().run(all_tests)
