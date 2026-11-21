def equation(t, P, params):
    s=1-np.exp(-params[4]*t)
    return params[0]+P*(params[1]+params[2]*s)+P**2*(params[3]+params[5]*s)+params[6]*P**3
