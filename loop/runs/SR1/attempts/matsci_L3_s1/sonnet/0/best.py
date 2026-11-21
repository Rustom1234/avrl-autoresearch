def equation(eps,T,params):
    F=eps**params[3]*np.exp(-eps*params[4])
    return params[0]+params[1]*T+params[2]*np.exp(-params[5]*T)*F+params[6]*np.exp(-params[7]*T)*eps