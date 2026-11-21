def equation(t,P,params):
    return -params[0]*np.sin(params[1]*t+params[2]) + params[3]*P + params[4]*P**2 + params[5]*np.log(P)