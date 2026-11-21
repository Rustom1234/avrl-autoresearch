def equation(t,P,params):
    return params[0]*P*(1-P/params[1]) - params[2]*np.tanh(params[3]*t)*(1+params[4]*P) + params[5]