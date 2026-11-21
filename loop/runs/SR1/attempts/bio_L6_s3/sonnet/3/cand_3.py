def equation(t,P,params):
    return params[0]*P*(1-P/params[1]) - params[2]*(1-np.exp(-t/params[3]))*P/(params[4]+P) - params[5]*P