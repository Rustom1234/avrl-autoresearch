def equation(t, P, params):
    return (params[0]*np.exp(-params[1]*t)+params[2])*P + params[3]*P*P - params[4]*np.log(t) - params[5]