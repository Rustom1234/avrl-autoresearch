def equation(eps, T, params):
    return params[0] + params[1]*np.sqrt(T)*eps**params[2] + params[3]*T*eps**params[4] + params[5]*eps**params[6]
