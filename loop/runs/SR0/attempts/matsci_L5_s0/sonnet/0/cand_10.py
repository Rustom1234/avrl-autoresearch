def equation(eps,T,params):
    return params[0]-params[1]*np.tanh(params[2]*eps)-params[3]*eps**2+params[4]*eps