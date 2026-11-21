def equation(eps, T, params):
    return params[0] + params[1]*np.log(eps) + params[2]*T*eps**params[3] + params[4]*np.exp(params[5]*eps*T) + params[6]*np.log(eps)**2 + params[7]*T