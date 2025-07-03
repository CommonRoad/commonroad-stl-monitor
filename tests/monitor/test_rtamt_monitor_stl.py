from copy import deepcopy
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.rule.rule_node import IOType


class TestRtamtStlMonitor:
    def test_evaluate_monitor_online(self):
        """Test presence of issue https://github.com/nickovic/rtamt/issues/83 in the installed rtamt package version."""

        mon = RtamtStlMonitor(
            "once[0,0.1s] predicate",
            [("predicate", IOType.OUTPUT)],
            0.1,
        )

        rob = mon.evaluate_monitor_online(0, [("predicate", -1.0)])

        assert 0.0 > rob

        rob = mon.evaluate_monitor_online(1, [("predicate", -1.0)])
        assert 0.0 > rob

        rob = mon.evaluate_monitor_online(2, [("predicate", 1.0)])
        assert rob >= 0.0

        rob = mon.evaluate_monitor_online(3, [("predicate", -1.0)])
        assert rob >= 0.0, "Issue #83 of rtamt is present!"

        rob = mon.evaluate_monitor_online(4, [("predicate", -1.0)])
        assert 0.0 > rob

        rob = mon.evaluate_monitor_online(5, [("predicate", -1.0)])
        assert 0.0 > rob

    def setup_method(self):
        self.formula = "a and b"
        self.predicates = [("a", IOType.INPUT), ("b", IOType.INPUT)]
        self.dt = 1.0

    def test_offline_evaluation_correctness(self):
        monitor = RtamtStlMonitor(self.formula, self.predicates, self.dt)
        a = [1, 1, 0]
        b = [1, 0, 1]
        result = monitor.evaluate_monitor_offline([("a", a), ("b", b)])
        expected = [1.0, 0.0, 0.0]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_online_evaluation_correctness(self):
        monitor = RtamtStlMonitor(self.formula, self.predicates, self.dt)
        steps = [(("a", 1.0), ("b", 1.0)), (("a", 1.0), ("b", 0.0)), (("a", 0.0), ("b", 1.0))]
        expected = [1.0, 0.0, 0.0]
        results = [monitor.evaluate_monitor_online(t, list(vals)) for t, vals in enumerate(steps)]
        assert results == expected, f"Expected {expected}, got {results}"

    def test_copy_preserves_behavior(self):
        monitor = RtamtStlMonitor(self.formula, self.predicates, self.dt)
        monitor_copy = deepcopy(monitor)

        a = [1, 1, 0]
        b = [1, 0, 1]
        result_orig = monitor.evaluate_monitor_offline([("a", a), ("b", b)])
        result_copy = monitor_copy.evaluate_monitor_offline([("a", a), ("b", b)])

        assert result_orig == result_copy

    def test_reset_clears_internal_state(self):
        monitor = RtamtStlMonitor(self.formula, self.predicates, self.dt)
        _ = monitor.evaluate_monitor_online(0, [("a", -1.0), ("b", -1.0)])
        monitor.reset()
        result = monitor.evaluate_monitor_online(0, [("a", 1.0), ("b", 1.0)])
        assert result == 1.0  # Should reset cleanly
