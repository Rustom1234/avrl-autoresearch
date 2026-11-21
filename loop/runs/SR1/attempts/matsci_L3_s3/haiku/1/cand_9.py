def equation(eps, T, params):
    return params[0] + params[1]*T + (params[2] + params[3]*T + params[4]*T**2)*(1-np.exp(-(params[5]+params[6]*T)*eps))