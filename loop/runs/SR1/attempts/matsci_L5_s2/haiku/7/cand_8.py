def equation(eps, T, params):
    return params[0]*np.log(1+params[1]*eps)*T**params[2] + params[3]*eps**params[4]