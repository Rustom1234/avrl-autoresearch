def equation(t,P,params):
    w=params[7]
    return params[0]*P - params[1]*P**2 + params[2] + params[3]*t + params[4]*P*np.sin(w*t) + params[5]*P*np.cos(w*t) + params[6]*np.sin(w*t) + params[8]*np.cos(w*t) + params[9]*P*t
