import unittest

class TestImport(unittest.TestCase):
    def test_import(self):
        import sys
        for s in sys.path:
            print(s)
        from crmonitor.common.road_network import RoadNetwork
        import pycrccosy
        print(pycrccosy.__file__)

if __name__ == '__main__':
    unittest.main()
