def equation(eps, T, params):
    return (params[0] + params[1]*eps**params[2])*T**params[3] + params[4]*T**params[5]*eps**params[6]*np.exp(params[7]*T)