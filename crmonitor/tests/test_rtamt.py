import unittest

from monitors import mtl
from rtamt import STLDiscreteTimeSpecification, Language


def conv(l):
    return [1.0 if e else -1.0 for e in l]


class TestRTAMT(unittest.TestCase):
    @unittest.SkipTest
    def test_backends(self):
        """
        Test bugfix https://github.com/nickovic/rtamt/pull/68
        :return:
        """
        spec_str = "(a >= 1.0)"
        values = [("a", 2.0)]

        spec = STLDiscreteTimeSpecification(
            semantics="robustness", language=Language.PYTHON
        )

        spec.declare_var("a", "float")
        spec.spec = spec_str
        spec.parse()

        rob = spec.update(0, values)
        self.assertGreater(rob, 0.0)

        spec = STLDiscreteTimeSpecification(
            semantics="robustness", language=Language.CPP
        )

        spec.declare_var("a", "float")
        spec.spec = spec_str
        spec.parse()

        rob = spec.update(0, values)
        self.assertGreater(rob, 0.0)

    def test_prev(self):
        spec_str = "prev(a)"
        values = [False]

        spec = STLDiscreteTimeSpecification(
            semantics="robustness", language=Language.PYTHON
        )

        spec.declare_var("a", "float")
        spec.spec = spec_str
        spec.parse()

        rob = [spec.update(t, [("a", v)]) for t, v in enumerate(conv(values))]

        mon = mtl.monitor("pre(a)")
        mon_rob = [mon.update(a=a) for a in values]
        self.assertEqual(rob[0] >= 0, mon_rob[0])

    def test_once(self):
        spec_str = "once[0,2](a and prev(!a))"
        values = [True, True, True, True, True]

        spec = STLDiscreteTimeSpecification(
            semantics="robustness", language=Language.PYTHON
        )

        spec.declare_var("a", "float")
        spec.spec = spec_str
        spec.parse()

        rob = [spec.update(t, [("a", v)]) >= 0.0 for t, v in enumerate(conv(values))]

        mon = mtl.monitor("once[0,2](a and pre(!a))")
        mon_rob = [mon.update(a=a) for a in values]
        self.assertListEqual(rob, mon_rob)

    def test_trace(self):
        same_lane = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

        in_front = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ]

        safe_dist = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ]

        cut_in = [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

        same_lane_stl = conv(same_lane)
        in_front_stl = conv(in_front)
        safe_dist_stl = conv(safe_dist)
        cut_in_stl = conv(cut_in)

        spec_str = "((in_front and same_lane and not once[0, 15](cut_in and prev(not cut_in))) implies safe_dist)"

        spec = STLDiscreteTimeSpecification(
            semantics="robustness", language=Language.PYTHON
        )

        spec.declare_var("same_lane", "float")
        spec.declare_var("in_front", "float")
        spec.declare_var("safe_dist", "float")
        spec.declare_var("cut_in", "float")
        spec.spec = spec_str
        spec.parse()

        rob = [
            spec.update(
                t,
                [
                    ("same_lane", sl),
                    ("in_front", fr),
                    ("safe_dist", sd),
                    ("cut_in", ci),
                ],
            )
            >= 0.0
            for t, (sl, fr, sd, ci) in enumerate(
                zip(same_lane_stl, in_front_stl, safe_dist_stl, cut_in_stl)
            )
        ]

        mon = mtl.monitor(
            "((in_front && same_lane && !once[0, 15](cut_in && pre(not cut_in))) -> safe_dist)"
        )
        mon_rob = [
            mon.update(same_lane=sl, in_front=fr, safe_dist=sd, cut_in=ci)
            for t, (sl, fr, sd, ci) in enumerate(
                zip(same_lane, in_front, safe_dist, cut_in)
            )
        ]
        self.assertListEqual(rob, mon_rob)


if __name__ == "__main__":
    unittest.main()
