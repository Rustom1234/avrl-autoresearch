def equation(t, P, params):
    return -(params[0]*np.exp(params[1]*t)*P + params[2]*np.exp(params[3]*t)*P**2)