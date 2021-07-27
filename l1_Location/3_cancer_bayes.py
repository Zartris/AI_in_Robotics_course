p_cancer = 0.001
p_not_cancer = 1 - p_cancer

p_positive_given_cancer = 0.8
p_positive_given_not_cancer = 0.1

# P(A|B) = P(B|A)*P(A) / P(B)


# Lets compute probability of having cancer given a positive result:
# p(cancer | positive)

# p_hat(cancer | positive) = P(B|A) * P(A) = p_cancer * p_positive_given_cancer = 0.001 * 0.8 = 0.0008
p_hat_1 = p_cancer * p_positive_given_cancer
# p_hat(not cancer | positive) = P(B|A) * P(A) = p_not_cancer * p_positive_given_not_cancer = 0.999 * 0.1 = 0.0999
p_hat_2 = p_not_cancer * p_positive_given_not_cancer

# P(B) = sum of the probability of being in any given state given B which is positive here
# P(B) = alpha = sum(p_hat_1, p_hat_2) = 0.0008 + 0.0999 =  0.1007
# This is P(B) since p_hat_1 and p_hat_2 is the probability of seeing a positive result.
p_positive = sum([p_hat_1, p_hat_2])

# Which divided gives us the result:
p_of_cancer_given_positive = p_hat_1 / p_positive
print(p_of_cancer_given_positive)
