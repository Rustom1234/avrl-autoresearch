def equation(t, P, params):
    k = params[1]
    return (params[0]*np.exp(-k*t)+params[2])*P + params[3]*P*P + params[6]*P*P*np.exp(-k*t) - params[4]*np.log(t) - params[5] + params[7]*t