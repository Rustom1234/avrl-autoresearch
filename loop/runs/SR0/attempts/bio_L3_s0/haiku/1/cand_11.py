def equation(t, P, params):
    return params[0] + params[1]*t/P**params[2] + params[3]*P + params[4]*P*P + params[5]*t + params[6]*np.log(P)