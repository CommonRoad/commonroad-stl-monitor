import unittest
import numpy as np

from commonroad.geometry.shape import Polygon

from crmonitor.predicates.predicate_collection import ConstraintType, ConstraintRepresentation, Constraint


class TestConstraint(unittest.TestCase):

    def test_constraint_initialization(self):
        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        constraint = Constraint(axis, ConstraintRepresentation.INNER_BOUNDARY, value)

        self.assertListEqual(constraint.axis, axis)
        np.testing.assert_equal(value.vertices, constraint.value.vertices)
        self.assertEqual(ConstraintRepresentation.INNER_BOUNDARY.value, constraint.constraint_representation.value)
