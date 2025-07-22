"""
Oracle adapters for maximal clique finding.

This module provides a unified interface for different oracle solvers
used in the Motzkin-Straus based maximal clique finding algorithm.
"""

from .factory import OracleFactory
from .base import OracleAdapter, OracleConfig

__all__ = ["OracleFactory", "OracleAdapter", "OracleConfig"]