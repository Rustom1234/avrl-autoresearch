def equation(t, P, params):
    return params[0]*P + params[1]*P**2 + params[2]*P**3 + params[3]*P**1.5 + params[4]*P**1.5*t**2 + params[5]*np.abs(P)**params[6]