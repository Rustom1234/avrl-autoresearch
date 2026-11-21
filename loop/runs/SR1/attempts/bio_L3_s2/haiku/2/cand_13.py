def equation(t, P, params):
    w = params[9]
    return (params[0] + params[1]*np.sin(w*t + params[2]) + params[6]*t)*P + params[3]*P*P + params[4] + params[5]*t + params[7]*np.sin(w*t + params[8])
