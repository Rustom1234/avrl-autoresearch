def equation(t, A, params):
    return params[0]/t + params[1] + params[2]*np.log(t) + A**params[3]*(params[4]*np.log(t)+params[5]*t+params[6]*np.log(t)**2)