def equation(eps, T, params):
    return params[0]*T**params[1] + params[2]*eps**2*np.sqrt(T) + params[3]*np.sin(params[4]*T+params[5])
