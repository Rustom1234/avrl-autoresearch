def equation(t,P,params):
    return params[0]*np.tanh(t/params[1])+P*(params[2]+params[3]*np.sin(params[4]*t+params[5]))-params[6]*P**2