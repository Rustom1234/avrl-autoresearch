def equation(t, P, params):
    r = params[0] + params[1]*np.cos(params[2]*t) + params[3]*t
    return r*P - params[4]*P**2 + params[5]*t*P**2 + params[6]*np.sin(params[7]*t)*P**2
