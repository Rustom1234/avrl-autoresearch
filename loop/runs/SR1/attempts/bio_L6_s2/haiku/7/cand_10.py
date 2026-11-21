def equation(t, P, params):
    w=params[4]
    return params[0]*P + params[1]*P**2 + P*(params[2]*np.cos(w*t)+params[3]*np.sin(w*t)) + params[5]*P**3 + params[6]*t + params[7]*P**2*np.cos(w*t) + params[8]*P**2*np.sin(w*t) + params[9]*P*t