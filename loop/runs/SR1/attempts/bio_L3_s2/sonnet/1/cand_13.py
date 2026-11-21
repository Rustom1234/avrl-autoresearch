def equation(t,P,params):
    return (params[0]+params[1]*t**params[2])*(1+params[5]*P)+P*(params[3]+params[7]*np.sin(params[6]*t+3.14159))+params[4]*P*P