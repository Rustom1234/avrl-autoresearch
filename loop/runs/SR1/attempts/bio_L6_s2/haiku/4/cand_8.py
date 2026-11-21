def equation(t, P, params):
    e=np.exp(-params[3]*t)
    return P*(params[0]+params[1]*P+params[2]*P**2+params[4]*e+params[5]*P*e+params[6]*P**2*e)