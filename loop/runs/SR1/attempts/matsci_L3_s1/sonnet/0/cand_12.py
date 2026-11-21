def equation(eps,T,params):
    F=eps**params[0]*np.exp(-params[1]*eps)
    return params[2]+params[3]*T+params[4]/T+(params[5]+params[6]*T+params[7]/T)*F+params[8]*eps/T+params[9]*eps