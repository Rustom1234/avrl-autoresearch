def equation(t,P,params):
    return params[0]+params[1]*t+params[2]*P*np.exp(-t/params[3])+params[4]*t*P/(params[5]+P)+params[6]*P