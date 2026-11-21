def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*t*P + params[3]*np.log(P) + params[4]*np.exp(-P*params[5])