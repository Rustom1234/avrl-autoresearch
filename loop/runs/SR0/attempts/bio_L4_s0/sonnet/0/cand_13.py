def equation(t, P, params):
    return params[0] + params[1]*np.log(t+params[7]) + params[2]*P*np.exp(-params[3]*t) + params[4]*P*np.exp(-P/params[5]) + params[6]*P