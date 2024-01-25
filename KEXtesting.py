import metrics
import numpy as np
from sympy import *

A = Matrix([[-3, 5, 7],
            [2, 6, 4],
            [0, 2, 8]])

assert metrics.norm1(A) == 19

assert metrics.norminf(A) == 15

assert metrics.frobnorm(A) == (3**2 + 5**2 + 7**2 + 2**2 + 6**2 + 4**2 + 2**2 + 8**2)**.5


B = Matrix([[2, 4, 6],
            [5, 3, 3]])

assert metrics.norm2(B) == 90.02468383590426**.5 # Need to calculate this by hand too see if correct...
