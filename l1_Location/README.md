# AI_in_Robotics_course

Lesson 1

## Bayes Threorem:
``P(A|B) = P(B|A)*P(A) / P(B)``

We introduce a variable called p_bar which is the non-normalized posterior

```markdown
Z = Measurements
X = Position in grid

p_bar(X_i, Z) = P(Z|X_i) P(X_i)
alpha         = sum(p_bar(X|Z))
p(X_i|Z)      = (1/alpha) * p_bar(X_i|Z)
```
This is exactly what we programmed in 2_location_robot_example.py in the method sense and see cancer example

## Total probability (motion)
https://classroom.udacity.com/courses/cs373/lessons/48739381/concepts/485326080923
```markdown
i = grid cell
t = time step
P(X_i^t) = sum_j(P(X_j^(t-1) * P(X_i | X_j))
```
what this means is the probability of being in cell ``X_i`` at time ``t`` is the ``sum`` of the probability of have been in cell ``X_j`` at timestep ``t-1`` times the probability of moving to cell ``X_i`` given we been in cell ``X_j``.

This is also written as **Theorem of total probability**:``P(A) = Sum_B P(A|B) P(B)``



## Test two coins:
Problem description
```markdown
fair coin: P(heads) = 0.5
loaded coin: P(heads) = 0.1
take coin with 50% chance to be fair, we flip it and observe H. 
what is the P(fair | heads)
```


We solve this with bayes theorem:
```markdown
p_bar(X_i, Z) <-- P(Z|X_i) P(X_i)
p_bar_1(fair | heads)   = P(heads | fair) * p(fair) = 0.5 * 0.5 = 0.25
p_bar_2(loaded | heads) = P(heads | loaded) * p(loaded) = 0.1 * 0.5 = 0.05
alpha                   = sum(p_bar_1, p_bar_2) = 0.25 + 0.05 = 0.3
P(fair | heads)         = p_hat_1 / alpha = 0.25 / 0.3 = 0.8333...
```


## homework:
### Probability

```markdown
Theorem of total probability:
--------------------------------------
Given: 
P(X) = 0.2
P(Y|X) = 0.6
P(Y|not X) = 0.6

What is P(Y) using P(X_i^t) = sum_j(P(X_j^(t-1)) * P(X_i | X_j))

P(Y) = sum_j(P(X) * P(Y | X), P(not X) * P(Y | not X))
     = P(X) * P(Y | X) + P(not X) * P(Y | not X)
     = 0.2 * 0.6 + 0.8 * 0.6
     = 0.6
```

### Bayes Threorem
```markdown
Bayes Threorem:
--------------------------------------
Given: 
P(F) = 0.001
B = yes it burns
P(lies) = 0.1

What is:
p_bar(F|B)     = P(B|F) * P(F) = 0.9 * 0.001 = 0.0009

p_bar(not F|B) = P(B|not F) * P(not F) = 0.1 * (1- 0.001) = 0.0999

alpha          = sum(p_bar(F|B), p_bar(not F|B)) = 0.0009 + 0.0999 = 0.1008

p(F|B)         = p_bar(F|B) / alpha = 0.0009 / 0.1008 = 0.00892

p(not F|B)     = p_bar(not F|B) / alpha = 0.0999 / 0.1008 = 0.991
```
