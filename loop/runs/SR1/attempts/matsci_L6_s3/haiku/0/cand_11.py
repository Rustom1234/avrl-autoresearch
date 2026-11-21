def equation(eps, T, params):
    A=np.abs(eps)
    return -(params[0]*np.exp(-params[1]*T) + params[2]*T + params[3]*A**params[4]*(1+params[5]*T) + params[6]*A)
