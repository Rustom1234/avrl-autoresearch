def equation(eps, T, params):
    return params[0]*np.cos(params[1]*eps + params[2]) + params[3] + params[4]*T*np.cos(params[5]*eps+params[6])
