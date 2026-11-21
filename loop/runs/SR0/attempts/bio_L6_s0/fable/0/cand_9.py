def equation(t, P, params):
    return params[0]*t**1.5+params[1]*np.sqrt(t)*P**3+params[2]*P**params[4]*(1-P/params[3])+params[5]*P