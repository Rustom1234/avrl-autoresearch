def equation(eps, T, params):
    return (params[0] + params[1]*T) + (params[2] + params[3]*T + params[6]*T**2)*(1-np.exp(-params[4]*eps**params[5]))