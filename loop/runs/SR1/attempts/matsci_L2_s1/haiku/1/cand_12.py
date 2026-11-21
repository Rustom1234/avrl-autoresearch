def equation(eps, T, params):
    return params[0] + params[1]*np.log(T) + params[2]*T + params[3]*np.sqrt(T)*eps**params[4] + params[5]*eps*T