import numpy as np

RED = 2
GREEN = 3

# n = input("N number of positions, where N is: ")
# initialize world
n = 5
color_grid = np.full((n), 3)
color_grid[1] = RED
color_grid[2] = RED

p = np.full((n), 1. / n)
pHit = 0.6
pMiss = 0.2
print(p)


# A function to update our believes given a new measurements
def sense(p, Z):
    q = p
    for i in range(len(p)):
        hit = color_grid[i] == Z
        q[i] *= (hit * pHit + (1 - hit) * pMiss)

    # Normalize
    s = np.sum(q)
    for i in range(len(q)):
        q[i] /= s
    return q


measurements = [RED, GREEN]
for m in measurements:
    p = sense(p, m)

print("sense:", p)
