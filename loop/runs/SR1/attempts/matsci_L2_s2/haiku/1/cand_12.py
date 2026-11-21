def equation(eps, T, params):
    return params[0] + params[1]*np.log(eps) + params[2]*np.log(eps)**2 + params[3]*np.exp(params[4]*eps*T) + params[5]*T + params[6]*eps*T