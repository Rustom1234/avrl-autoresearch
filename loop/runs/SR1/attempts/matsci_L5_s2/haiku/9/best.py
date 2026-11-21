def equation(eps, T, params):
    return params[0]*eps**params[1]/(1+params[2]*eps**params[3]) + params[4]*np.exp(params[5]*eps)*T**params[6] + params[7]*T*eps**params[8]