def equation(t, P, params):
    w=params[5]
    return params[0]*P+params[1]*P**2+params[2]/P+params[3]+params[4]*np.sin(w*t)+params[6]*np.cos(w*t)+params[7]*P**3+params[8]*P*np.sin(w*t)+params[9]*P*np.cos(w*t)
