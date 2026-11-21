def equation(t, A, params):
    return (params[0]*A**params[1] + params[2]*A**params[3]*t) / (1 + params[4]*t) + params[5]*A**params[6]*t*np.exp(-params[7]*t)