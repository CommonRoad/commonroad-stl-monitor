import numpy as np


class Polynomial:
    @classmethod
    def from_boundary(
        cls,
        left: list[float | None],
        right: list[float | None],
        interval: float,
    ) -> np.polynomial.Polynomial:
        """generate polynomial function y=f(x) with boundary conditions

        Args:
            left (list[Optional[ float]]): left boundary conditions,
                [y,y',y'',...] | x=0, None means free.
            right (list[Optional[ float]]): right boundary conditions,
                [y,y',y'',...] | x=t, None means free.
            interval (float): t.

        Returns:
            numpy.polynomial.Polynomial: the polynomial function.
        """
        degree = sum(x is not None for x in left + right) - 1
        order = max(len(left), len(right)) - 1
        t = cls.powers(interval, degree)
        derivative_parameter = cls.get_derivative_parameter(degree, order)
        A, b = cls.get_linear_system(derivative_parameter, left, right, t)
        coef = np.linalg.solve(A, b)
        return np.polynomial.Polynomial(coef)

    @staticmethod
    def get_derivative_parameter(degree: int, order: int) -> np.ndarray:
        """get the matrix containing parameters of derivative of order [0 to order].
        The i-th order derivative of polynomial function at x is calculated by multiplying
        the i-th row of the matrix with transpose of powers(x,degree).

        Args:
            degree (int): degree of the polynomial
            order (int): max necessary order of the derivatives

        Returns:
            numpy.ndarray: the derivative parameter matrix
        """
        res = np.ones((1, degree + 1))
        if order == 0:
            return res

        multiplier = np.arange(degree + 1).reshape((1, -1))
        for i in range(order):
            res = np.concatenate([res, res[i] * multiplier], axis=0)
            multiplier[multiplier > 0] -= 1

        return res

    @staticmethod
    def powers(x: float, n: int) -> np.ndarray:
        """get the array containing powers of x

        Args:
            x (float): base
            n (int): max exponent

        Returns:
            numpy.ndarray: powers of x, [x^0, x^1, ..., x^n]
        """
        res = [1]
        for i in range(n):
            res.append(res[i] * x)
        return np.array(res)

    @classmethod
    def get_linear_system(
        cls,
        derivative_parameter: np.ndarray,
        left: list[float | None],
        right: list[float | None],
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """construct the linear system to solve the coefficients of the polynomial function

        Args:
            derivative_parameter (numpy.ndarray): derivative parameter matirx
            left (list[Optional[ float]]): left boundary conditions
            right (list[Optional[ float]]): right boundary conditions
            t (numpy.ndarray): powers of t

        Returns:
            Tuple[numpy.ndarray,numpy.ndarray]: A, b for linear system A * x = b
        """
        A_left, b_left = cls.get_linear_system_left(derivative_parameter, left)
        A_right, b_right = cls.get_linear_system_right(derivative_parameter, right, t)
        A = np.concatenate([A_left, A_right], axis=0)
        b = np.concatenate([b_left, b_right], axis=0)
        return A, b

    @classmethod
    def get_linear_system_left(
        cls, derivative_parameter: np.ndarray, left: list[float | None]
    ) -> tuple[np.ndarray, np.ndarray]:
        """construct half the linear system considering the left condition

        Args:
            derivative_parameter (numpy.ndarray): derivative parameter matirx
            left (list[Optional[ float]]): left boundary conditions

        Returns:
            Tuple[numpy.ndarray,numpy.ndarray]: A, b for the half linear system
        """
        diag = np.diag(derivative_parameter)
        A = np.zeros(derivative_parameter.shape)
        np.fill_diagonal(A, diag)
        b = np.array(left).reshape([-1, 1])
        return cls.drop_none(A, b)

    @classmethod
    def get_linear_system_right(
        cls,
        derivative_parameter: np.ndarray,
        right: list[float | None],
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """construct half the linear system considering the right condition

        Args:
            derivative_parameter (numpy.ndarray): derivative parameter matirx
            right (list[Optional[ float]]): right boundary conditions
            t (numpy.ndarray): powers of t

        Returns:
            Tuple[numpy.ndarray,numpy.ndarray]: A, b for the half linear system
        """

        A = derivative_parameter.copy()
        for i, row in enumerate(A):
            row *= np.roll(t, i)
        b = np.array(right, dtype=object).reshape([-1, 1])
        return cls.drop_none(A, b)

    @staticmethod
    def drop_none(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """clean the linear system by dropping the columns containing free boundary conditions

        Args:
            A (numpy.ndarray): A for the linear system
            b (numpy.ndarray): b for the linear system

        Returns:
            Tuple[numpy.ndarray,numpy.ndarray]: A, b for the cleaned linear system
        """
        b = b.astype("float64")
        A = A[0 : len(b), :]
        A = A[(~np.isnan(b)).flatten(), :]
        b = b[~np.isnan(b)]
        return A, b
