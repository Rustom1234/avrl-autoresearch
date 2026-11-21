def equation(eps, T, params):
    return params[0]*T**params[1] + eps**2*T**params[2]*(params[3] + params[4]*np.cos(params[5]*T) + params[6]*np.sin(params[5]*T)) + params[7]*np.sin(params[8]*T+params[9])
