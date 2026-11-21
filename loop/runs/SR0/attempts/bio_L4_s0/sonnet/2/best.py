def equation(t,P,params):
    return params[0]*P/(params[1]+P)+params[2]*t+params[3]*np.sin(params[4]*t+params[5])*P/(params[6]+P)+params[7]*P