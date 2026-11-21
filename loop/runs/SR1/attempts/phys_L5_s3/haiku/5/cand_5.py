def equation(t, x, v, params):
    return (params[0] + params[1]*t)*(1-np.exp(-params[2]*x))*np.exp(-params[3]*x) + params[4]*v + params[5]