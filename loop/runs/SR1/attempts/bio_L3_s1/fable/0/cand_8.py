def equation(t, P, params):
    return params[0]*np.abs(P)**params[1] + params[2]*P + (params[3]+params[4]*t)*P*(1-np.abs(P/params[5])**params[6])+params[7]