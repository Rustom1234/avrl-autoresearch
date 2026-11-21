def equation(eps, T, params):
    return params[0]*eps**params[1]*(1+params[2]*eps)**params[3]*np.exp(params[4]*T) + params[5]*eps**params[6]*T + params[7]*np.sqrt(eps)*np.exp(params[8]*T)+params[9]
