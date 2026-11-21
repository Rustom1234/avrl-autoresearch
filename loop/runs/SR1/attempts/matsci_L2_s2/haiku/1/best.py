def equation(eps, T, params):
    L=np.log(eps)
    return params[0] + params[1]*L + params[2]*L**2 + params[3]*T*eps**params[4] + params[5]*np.exp(params[6]*eps*T) + params[7]*T + params[8]*T*L