def equation(eps, T, params):
    return (params[0]+params[1]*T**2)*eps**params[2]*np.exp(params[3]*eps/T) + params[4]*eps*T**params[5] + params[6]*eps**2*T**params[7]