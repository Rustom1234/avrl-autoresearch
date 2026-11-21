def equation(t,P,params):
    return params[0]/P + P*(params[1]+params[2]*np.sin(params[3]*t+params[4])) - params[5]*P**2 + params[6]