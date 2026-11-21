def equation(eps, T, params):
    return -(params[0] + params[1]/T + params[2]*(eps**params[3])*(1+params[4]/T) + params[5]*np.exp(-params[6]*T))
