def equation(t,P,params):
    return params[0]*P/(params[1]+P) - params[2]*t*np.exp(-params[3]*P) + params[4]*np.sqrt(P)*t