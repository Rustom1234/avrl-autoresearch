def equation(t,P,params):
    return P*(params[0]+params[1]*t+params[2]*np.sin(params[3]*t+params[4]))-params[5]*P**2+params[6]*np.exp(-t)*P