def equation(t,P,params):
    return (params[0]+params[1]*t+params[7]*t**2 + P*(params[2]+params[3]*np.sin(params[4]*t+params[5])) + P**2*(params[6]+params[8]*t))
