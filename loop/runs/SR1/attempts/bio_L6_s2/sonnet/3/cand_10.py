def equation(t,P,params):
    return -P*(params[0]+params[1]*np.sqrt(t)+params[2]*P**params[3]+params[4]*np.sqrt(t)*P)