def equation(t,P,params):
    return params[0]*(1-(P/(params[1]*(1+params[2]*t)))**params[3]) + params[4]*np.log(t+params[5])