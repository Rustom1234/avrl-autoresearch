def equation(eps, T, params):
    return -(params[0]*T**params[1] + params[2]*(1-np.exp(-params[3]*eps))*T**params[4] + params[5]*eps**params[6])