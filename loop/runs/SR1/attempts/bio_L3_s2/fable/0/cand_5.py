def equation(t, P, params):
    return params[0]*P*(1-P/params[1]) + params[2]*P*np.sin(params[3]*t+params[4]) + params[5]*np.abs(t)**params[6] + params[7]*P/(np.abs(params[8])+P)+params[9]