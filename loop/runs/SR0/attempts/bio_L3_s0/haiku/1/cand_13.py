def equation(t, P, params):
    return params[0] + params[1]*t/P + params[2]*P + params[3]*P*P + params[4]*np.log(P)