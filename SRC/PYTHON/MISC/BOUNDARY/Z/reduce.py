import numpy as np
from sum import compute_sum
from itertools import combinations

n = 20

S = []

l = n // 2 - 1 
u = n - 2

# ----- A.1
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 1, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("A.1", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.2
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 3, 1, 2], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("A.2", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.3
lo  = np.array([1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([l, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 2, 3, 1, 3, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 2, 1], dtype=np.int32)

denom_sign = np.array([-1, 0, 0, 1], dtype=np.int32)

S.append(("A.3", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.4
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 2, 1, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("A.4", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.5
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 1, 2], dtype=np.int32)

denom_sign = np.array([0, 1, -1, 0], dtype=np.int32)

S.append(("A.5", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.6
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("A.6", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.7
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 1, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("A.7", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.8
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 2, 1], dtype=np.int32)

denom_sign = np.array([-1, 0, 0, 1], dtype=np.int32)

S.append(("A.8", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.9
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("A.9", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.10
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([1, 0, 2, 1], dtype=np.int32)

denom_sign = np.array([-1, 0, 0, 1], dtype=np.int32)

S.append(("A.10", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.11
lo  = np.array([l+1, l+1, 1, 1], dtype=np.int32)
hi  = np.array([u, u, l, l], dtype=np.int32)

sin_k = np.array([1, 2, 3, 2, 3, 3, 1, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 1, 2], dtype=np.int32)

denom_sign = np.array([0, 1, -1, 0], dtype=np.int32)

S.append(("A.11", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.12
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([0, 1, 1, 2], dtype=np.int32)

denom_sign = np.array([0, 1, -1, 0], dtype=np.int32)

S.append(("A.12", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- A.13
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 3, 1, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("A.13", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.1
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 1, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("B.1", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.2
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 3, 1, 2], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("B.2", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.3
lo  = np.array([1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([l, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 2, 3, 3, 1, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 2, 1], dtype=np.int32)

denom_sign = np.array([0, -1, 0, 1], dtype=np.int32)

S.append(("B.3", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.4
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 3, 1, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("B.4", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.5
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 2, 3, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 1, 2], dtype=np.int32)

denom_sign = np.array([1, 0, -1, 0], dtype=np.int32)

S.append(("B.5", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.6
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([2, 0, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("B.6", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.7
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 1, 3, 2], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("B.7", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.8
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 2, 1], dtype=np.int32)

denom_sign = np.array([0, -1, 0, 1], dtype=np.int32)

S.append(("B.8", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.9
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 3, 2, 3, 2], dtype=np.int32)

phase_pow = np.array([2, 0, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("B.9", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.10
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([0, 1, 2, 1], dtype=np.int32)

denom_sign = np.array([0, -1, 0, 1], dtype=np.int32)

S.append(("B.10", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.11
lo  = np.array([l+1, l+1, 1, 1], dtype=np.int32)
hi  = np.array([u, u, l, l], dtype=np.int32)

sin_k = np.array([1, 2, 3, 2, 3, 3, 1, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 1, 2], dtype=np.int32)

denom_sign = np.array([1, 0, -1, 0], dtype=np.int32)

S.append(("B.11", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.12
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([1, 0, 1, 2], dtype=np.int32)

denom_sign = np.array([1, 0, -1, 0], dtype=np.int32)

S.append(("B.12", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- B.13
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 3, 1, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("B.13", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.1
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 1, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("C.1", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.2
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 3, 2, 1], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("C.2", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.3
lo  = np.array([1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([l, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 2, 3, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 2, 1], dtype=np.int32)

denom_sign = np.array([-1, 0, 0, 1], dtype=np.int32)

S.append(("C.3", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.4
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 1, 2, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("C.4", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.5
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 2, 3, 1, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 1, 2], dtype=np.int32)

denom_sign = np.array([0, 1, -1, 0], dtype=np.int32)

S.append(("C.5", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.6
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("C.6", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.7
lo  = np.array([1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([l, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 3, 3, 1], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([-1, 1, -1, 1], dtype=np.int32)

S.append(("C.7", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.8
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 2, 1], dtype=np.int32)

denom_sign = np.array([-1, 0, 0, 1], dtype=np.int32)

S.append(("C.8", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.9
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 3, 3, 1], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("C.9", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.10
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 2, 3, 3, 3, 3, 1], dtype=np.int32)

phase_pow = np.array([1, 0, 2, 1], dtype=np.int32)

denom_sign = np.array([-1, 0, 0, 1], dtype=np.int32)

S.append(("C.10", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.11
lo  = np.array([l+1, l+1, 1, 1], dtype=np.int32)
hi  = np.array([u, u, l, l], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 1, 2], dtype=np.int32)

denom_sign = np.array([0, 1, -1, 0], dtype=np.int32)

S.append(("C.11", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.12
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([0, 1, 1, 2], dtype=np.int32)

denom_sign = np.array([0, 1, -1, 0], dtype=np.int32)

S.append(("C.12", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- C.13
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("C.13", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.1
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 1, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("D.1", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.2
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 3, 2, 1], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("D.2", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.3
lo  = np.array([1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([l, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 2, 1], dtype=np.int32)

denom_sign = np.array([0, -1, 0, 1], dtype=np.int32)

S.append(("D.3", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.4
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 1, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("D.4", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.5
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 2, 1, 3, 3, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 1, 2], dtype=np.int32)

denom_sign = np.array([1, 0, -1, 0], dtype=np.int32)

S.append(("D.5", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.6
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 2, 3, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("D.6", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.7
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 3, 2, 3, 1], dtype=np.int32)

phase_pow = np.array([1, 1, 1, 1], dtype=np.int32)

denom_sign = np.array([1, -1, -1, 1], dtype=np.int32)

S.append(("D.7", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.8
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 1, 3, 3, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([0, 1, 2, 1], dtype=np.int32)

denom_sign = np.array([0, -1, 0, 1], dtype=np.int32)

S.append(("D.8", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.9
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 3, 2, 3, 3, 1], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("D.9", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.10
lo  = np.array([1, 1, l+1, l+1], dtype=np.int32)
hi  = np.array([l, l, u, u], dtype=np.int32)

sin_k = np.array([1, 2, 2, 3, 3, 3, 3, 1], dtype=np.int32)

phase_pow = np.array([0, 1, 2, 1], dtype=np.int32)

denom_sign = np.array([0, -1, 0, 1], dtype=np.int32)

S.append(("D.10", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.11
lo  = np.array([l+1, l+1, 1, 1], dtype=np.int32)
hi  = np.array([u, u, l, l], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([1, 0, 1, 2], dtype=np.int32)

denom_sign = np.array([1, 0, -1, 0], dtype=np.int32)

S.append(("D.11", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.12
lo  = np.array([l+1, l+1, 1, l+1], dtype=np.int32)
hi  = np.array([u, u, l, u], dtype=np.int32)

sin_k = np.array([1, 2, 3, 1, 3, 3, 3, 2], dtype=np.int32)

phase_pow = np.array([1, 0, 1, 2], dtype=np.int32)

denom_sign = np.array([1, 0, -1, 0], dtype=np.int32)

S.append(("D.12", -compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

# ----- D.13
lo  = np.array([l+1, 1, 1, l+1], dtype=np.int32)
hi  = np.array([u, l, l, u], dtype=np.int32)

sin_k = np.array([1, 1, 3, 3, 2, 3, 2, 3], dtype=np.int32)

phase_pow = np.array([0, 2, 1, 1], dtype=np.int32)

denom_sign = np.array([0, 0, -1, 1], dtype=np.int32)

S.append(("D.13", compute_sum(n, lo, hi, sin_k, phase_pow, denom_sign)))

tol = 1e-10

nonzero = []
zero_terms = []

for label, val in S:
	if abs(val) < tol:
		zero_terms.append(label)
	else:
		nonzero.append((label, val))

print("\nVANISHING\n")
for l in zero_terms:
	print(l)

remaining = nonzero.copy()
used = set()
cancelled = []

for i, (label_a, a) in enumerate(nonzero):
	if i in used:
		continue
	for j, (label_b, b) in enumerate(nonzero):
		if j <= i or j in used:
			continue
		if abs(a + b) < tol:
			used.add(i)
			used.add(j)
			cancelled.append((label_a, label_b))
			break

remaining = [(l, v) for k, (l, v) in enumerate(nonzero) if k not in used]

print("\nCANCELLED\n")
for label_a, label_b in cancelled:
	print(f"{label_a} cancels {label_b}")

groups = []
used = set()

for i, (label_a, a) in enumerate(remaining):
	if i in used:
		continue

	group = [label_a]
	used.add(i)

	for j, (label_b, b) in enumerate(remaining):
		if j <= i or j in used:
			continue
		if abs(a - b) < tol:
			group.append(label_b)
			used.add(j)

	if len(group) > 1:
		groups.append(group)

print("\nEQUAL\n")
for g in groups:
	print(g)
