def equation(t,P,params):
    return P*(params[0]-params[1]*np.log(P))-(params[2]+params[3]*t+params[4]*t**2+params[5]*t**3)/(P+params[6])+params[7]*P*np.sin(params[8]*t+params[9])