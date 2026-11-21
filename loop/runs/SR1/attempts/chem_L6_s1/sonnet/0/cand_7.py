def equation(t, A, params):
    base = params[0]*A + params[1]*A**2 + params[2] + t*(params[3] + params[4]*A + params[5]*A**2)
    return base + params[6]*t**2*np.exp(-params[7]*A) + params[8]*t**2