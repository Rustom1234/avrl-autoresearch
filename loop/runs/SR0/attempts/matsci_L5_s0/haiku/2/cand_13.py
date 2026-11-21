def equation(eps, T, params):
    return params[0]*np.tanh(params[1]*(params[2]-eps)) + params[3] + params[4]*eps + params[5]*eps**2 + params[6]*T
