def equation(t,P,params):
    return params[0]*P*(1-(np.abs(P)/params[1])**params[2])