def equation(eps, T, params):
    return params[0]*np.exp(params[1]/T)*(eps+params[2]*np.exp(params[3]*T))**params[4] + params[5]*eps**2
