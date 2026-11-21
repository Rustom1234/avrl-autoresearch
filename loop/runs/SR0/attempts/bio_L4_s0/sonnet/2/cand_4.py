def equation(t,P,params):
    return (params[0]+params[1]*t+params[2]*np.exp(-params[3]*t))*P/(params[4]+P)+params[5]*P**params[6]