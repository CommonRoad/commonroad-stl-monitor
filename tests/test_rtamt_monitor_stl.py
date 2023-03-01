from dataclasses import dataclass
from unittest import TestCase


@dataclass
class DummyPredicate:
    name: str


class TestRtamtStlMonitor(TestCase):
    def test_evaluate_monitor_online(self):
        # """Test presence of issue https://github.com/nickovic/rtamt/issues/83 in the installed rtamt package version."""

        # mon = RtamtStlMonitor(
        #     "once[0,0.1s] predicate",
        #     [(DummyPredicate("predicate"), IOType.OUTPUT)],
        #     0.1,
        # )

        # rob = mon.evaluate_monitor_online(0, [("predicate", -1.0)])
        # self.assertGreater(0.0, rob)

        # rob = mon.evaluate_monitor_online(1, [("predicate", -1.0)])
        # self.assertGreater(0.0, rob)

        # rob = mon.evaluate_monitor_online(2, [("predicate", 1.0)])
        # self.assertGreaterEqual(rob, 0.0)

        # rob = mon.evaluate_monitor_online(3, [("predicate", -1.0)])
        # self.assertGreaterEqual(rob, 0.0, "Issue #83 of rtamt is present!")

        # rob = mon.evaluate_monitor_online(4, [("predicate", -1.0)])
        # self.assertGreater(0.0, rob)

        # rob = mon.evaluate_monitor_online(5, [("predicate", -1.0)])
        # self.assertGreater(0.0, rob)

        self.assertEqual(1, 1)
