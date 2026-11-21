def equation(eps, T, params):
    return params[0]*eps**params[1]*np.exp(params[2]*T + params[3]*eps + params[4]*T*np.log(eps))*T**params[5]