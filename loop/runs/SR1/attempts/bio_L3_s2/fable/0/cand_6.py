def equation(t, P, params):
    return params[0]*P*(1-P/params[1]) + params[2]*P*np.sin(params[3]*t+params[4]) + params[5]*np.log(1+np.abs(params[6])*t) + params[7]*np.abs(P)**params[8]+params[9]