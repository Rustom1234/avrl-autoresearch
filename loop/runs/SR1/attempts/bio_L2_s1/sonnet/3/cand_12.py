def equation(t,P,params):
    K=params[1]+params[2]*np.sqrt(t)
    return (params[0]+params[3]*np.sqrt(t))*(1-(P/K)**params[4])+params[5]*P