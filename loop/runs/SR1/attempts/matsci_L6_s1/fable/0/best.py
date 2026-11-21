def equation(eps, T, params):
    return params[0]*eps**3/T + params[1]/T + params[2]*T*np.log(T)