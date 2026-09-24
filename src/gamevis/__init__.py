"""gamevis: pictures of learning dynamics in games.

Modules
-------
simplex      first-order learning rules on probability simplices
dynamics     fixed-step integrators for tuples of simplices
signaling    sender-receiver signaling games (Lewis, costly signaling)
normal_form  2x2 bimatrix games and symmetric n-strategy games
markets      Bertrand, Cournot and Edgeworth duopolies
"""

from . import dynamics, markets, normal_form, signaling, simplex

__all__ = ["dynamics", "markets", "normal_form", "signaling", "simplex"]
__version__ = "0.1.0"
