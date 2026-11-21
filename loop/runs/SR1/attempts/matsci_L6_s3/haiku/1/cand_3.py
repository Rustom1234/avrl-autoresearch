def equation(eps, T, params):
    return -(params[0] + params[1]*T**params[2] + params[3]*(1-np.exp(-params[4]*eps))*(1+params[5]*T))
