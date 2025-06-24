"""
Custom exceptions for the Motzkin-Straus MIS solver.
"""


class MotzkinStrausError(Exception):
    """Base exception for Motzkin-Straus solver errors."""
    pass


class OracleError(MotzkinStrausError):
    """Raised when an oracle fails to solve the quadratic program."""
    pass


class SolverUnavailableError(MotzkinStrausError):
    """Raised when a required solver is not available or not installed."""
    pass