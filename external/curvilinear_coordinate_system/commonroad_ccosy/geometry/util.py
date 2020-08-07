from typing import Union, List
import numpy as np
import enum


@enum.unique
class Orientation(enum.Enum):
    """ Enum representing the orientation of a set of points."""
    COLINEAR = 0
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2


def intersection_line_line(l1_pt1: Union[List, np.ndarray], l1_pt2: Union[List, np.ndarray],
                           l2_pt1: Union[List, np.ndarray], l2_pt2: Union[List, np.ndarray]) \
        -> Union[None, np.ndarray]:
    """
    Intersection between two lines, defined by four points.
    Code from http://www.ahristov.com/tutorial/geometry-games/intersection-lines.html

    :param l1_pt1: start point of first line
    :param l1_pt2: end point of first line
    :param l2_pt1: start point of second line
    :param l2_pt2: end point of second line
    :return: if lines intersect their intersection point is returned; otherwise None
    """
    x1 = l1_pt1[0]
    y1 = l1_pt1[1]
    x2 = l1_pt2[0]
    y2 = l1_pt2[1]
    x3 = l2_pt1[0]
    y3 = l2_pt1[1]
    x4 = l2_pt2[0]
    y4 = l2_pt2[1]

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if np.abs(d) <= 0.000001:
        intersection_point = None
    else:
        xi = ((x3 - x4) * (x1 * y2 - y1 * x2) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        yi = ((y3 - y4) * (x1 * y2 - y1 * x2) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        intersection_point = np.array([xi, yi])
    return intersection_point


def intersection_segment_segment(l1_pt1: np.ndarray, l1_pt2: np.ndarray, l2_pt1: np.ndarray, l2_pt2: np.ndarray) \
        -> Union[None, np.ndarray]:
    """
    if two segments intersect, the function return the intersection point; otherwise None is returned

    :param l1_pt1: start point of first segment
    :param l1_pt2: end point of first segment
    :param l2_pt1: start point of second segment
    :param l2_pt2: end point of second segment
    :return: intersection point if segments intersect; otherwise None
    """
    if segments_intersect(l1_pt1, l1_pt2, l2_pt1, l2_pt2):
        return intersection_line_line(l1_pt1, l1_pt2, l2_pt1, l2_pt2)
    else:
        return None


def determine_orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> Orientation:
    """
    determines if three points are ordered clockwise, counterclockwise, or colinear.
    Reference: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    :param p: first point
    :param q: second point
    :param r: third point
    :return: orientation of points
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if val > 0.0:
        return Orientation.CLOCKWISE
    elif val < 0.0:
        return Orientation.COUNTERCLOCKWISE
    else:
        return Orientation.COLINEAR


def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """
    Given three points p, q, r, the function checks if point q lies on the line segment 'pr'.

    :param p: start point of the segment
    :param q: point to be tested if it is on the segment
    :param r: end point of the segment
    :return: True, if point lies on segment; otherwise false
    """
    orientation = determine_orientation(p, q, r)
    if orientation is not Orientation.COLINEAR:
        return False
    return on_segment_given_colinear_points(p, q, r)


def on_segment_given_colinear_points(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """
    Given three colinear points p, q, r, the function checks if point q lies on the line segment 'pr', algorithm from:
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    :param p: start point of the segment
    :param q: point to be tested if it is on the segment
    :param r: end point of the segment
    :return: True, if point lies on segment; otherwise false
    """
    if (np.greater_equal(max(p[0], r[0]), q[0]) and
            np.greater_equal(q[0], min(p[0], r[0])) and
            np.greater_equal(max(p[1], r[1]), q[1]) and
            np.greater_equal(q[1], min(p[1], r[1]))):
        return True
    else:
        return False


def segments_intersect(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray):
    """
    checks if two segments intersect, algorithm from:
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    :param p1: start point of first segment
    :param q1: end point of first segment
    :param p2: start point of second segment
    :param q2: end point of second segment
    :return:
    """
    o1 = determine_orientation(p1, q1, p2)
    o2 = determine_orientation(p1, q1, q2)
    o3 = determine_orientation(p2, q2, p1)
    o4 = determine_orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == Orientation.COLINEAR and on_segment_given_colinear_points(p1, p2, q1):
        return True
    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if o2 == Orientation.COLINEAR and on_segment_given_colinear_points(p1, q2, q1):
        return True
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == Orientation.COLINEAR and on_segment_given_colinear_points(p2, p1, q2):
        return True
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == Orientation.COLINEAR and on_segment_given_colinear_points(p2, q1, q2):
        return True
    # Doesn't fall in any of the above cases
    return False


def chaikins_corner_cutting(polyline: np.ndarray):
    """
    Chaikin's corner cutting algorithm to smooth a polyline by replacing each original point with two new points.
    The new points are at 1/4 and 3/4 along the way of an edge.

    :param polyline: polyline with 2D points
    :return: smoothed polyline
    """
    new_polyline = list()
    new_polyline.append(polyline[0])
    for i in range(0, len(polyline) - 1):
        new_polyline.append((3 / 4) * polyline[i] + (1 / 4) * polyline[i + 1])
        new_polyline.append((1 / 4) * polyline[i] + (3 / 4) * polyline[i + 1])
    new_polyline.append(polyline[-1])
    return new_polyline


def resample_polyline(polyline: np.ndarray, step: int = 2.0):
    """
    Resamples point with equidistant spacing.

    :param polyline: polyline with 2D points
    :param step: sampling interval
    :return: resampled polyline
    """
    if len(polyline) < 2:
        return np.array(polyline)
    new_polyline = [polyline[0]]
    current_position = step
    current_length = np.linalg.norm(polyline[0] - polyline[1])
    current_idx = 0
    while current_idx < len(polyline) - 1:
        if current_position >= current_length:
            current_position = current_position - current_length
            current_idx += 1
            if current_idx > len(polyline) - 2:
                break
            current_length = np.linalg.norm(polyline[current_idx + 1]
                                            - polyline[current_idx])
        else:
            rel = current_position / current_length
            new_polyline.append((1 - rel) * polyline[current_idx] +
                                rel * polyline[current_idx + 1])
            current_position += step
    new_polyline.append(polyline[-1])
    return np.array(new_polyline)


def compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvature of a given polyline

    :param polyline: The polyline for the curvature computation
    :return: The curvature of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)
    x_d = np.gradient(polyline[:, 0])
    x_dd = np.gradient(x_d)
    y_d = np.gradient(polyline[:, 1])
    y_dd = np.gradient(y_d)

    # compute curvature
    curvature = (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))

    return curvature


def compute_pathlength_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the path length of a given polyline

    :param polyline: polyline with 2D points
    :return: path length of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for pathlenth computation p={}'.format(polyline)
    distance = [0]
    for i in range(1, len(polyline)):
        distance.append(distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1]))
    return np.array(distance)


def compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the orientation of a given polyline

    :param polyline: polyline with 2D points
    :return: orientation of polyline
    """
    assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(polyline[0, :]) == 2, \
        'not a valid polyline. polyline = {}'.format(polyline)

    if len(polyline) < 2:
        raise NameError('Cannot create orientation from polyline of length < 2')

    orientation = []
    for i in range(0, len(polyline) - 1):
        pt1 = polyline[i]
        pt2 = polyline[i + 1]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    for i in range(len(polyline) - 1, len(polyline)):
        pt1 = polyline[i - 1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    return np.array(orientation)
