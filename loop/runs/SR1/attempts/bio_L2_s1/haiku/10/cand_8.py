def equation(t, P, params):
    return (params[0] + params[1]*t + params[2]*t*P)*np.exp(-params[3]*P**params[4]) + params[5]*P*np.exp(-params[6]*t)