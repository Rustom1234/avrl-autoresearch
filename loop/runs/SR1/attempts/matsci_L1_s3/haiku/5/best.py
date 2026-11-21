def equation(eps, T, params):
    return -(params[0]*eps*np.exp(params[1]*T) + params[2]*eps**params[3]*np.exp(params[4]*T) + params[5]*eps**2*T)
