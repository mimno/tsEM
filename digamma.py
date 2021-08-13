from scipy.special import digamma
import numpy

coef_1 = 1 / (4 * 3 * 2)
coef_2 = 37 / (8 * 6 * 5 * 4 * 3 * 2)
coef_3 = 10313 / (72 * 8 * 7 * 5 * 4 * 3 * 2)
coef_4 = 5509121 / (384 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2)

def exp_digamma(x):
    x_minus = x - 0.5
    inv_x_minus = 1.0 / x_minus

    result = x + inv_x_minus * coef_1
    acc = inv_x_minus * inv_x_minus * inv_x_minus
    result -= acc * coef_2
    acc *= inv_x_minus * inv_x_minus
    result += acc * coef_3
    acc *= inv_x_minus * inv_x_minus
    result -= acc * coef_4
    return  result

for x in [0.2, 0.51, 1.0, 3.0, 10.0]:
    print(x, numpy.exp(digamma(x)), exp_digamma(x))