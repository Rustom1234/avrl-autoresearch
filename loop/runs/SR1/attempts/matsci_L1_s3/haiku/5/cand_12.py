def equation(eps, T, params):
    return -params[0]*eps**params[1]*np.exp(params[2]*T)*(1+params[3]*eps)**params[4]*(1+params[5]*T*eps)
