def equation(t, P, params):
    w = params[9]
    return (params[0] + params[1]*np.sin(w*t + params[2]))*P + params[3]*P*P + params[4] + params[5]*t + params[6]*np.sin(w*t + params[7]) + params[8]*P*np.cos(w*t)
