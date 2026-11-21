def equation(t, P, params):
    return params[0]*t**1.5+params[1]*np.sqrt(t)*P**3+params[2]*P*(1-P/params[3])*(1-np.exp(-P/params[4]))