def equation(eps, T, params):
    return -(params[0] + params[1]*np.abs(eps)**params[2])*np.exp(-params[3]*T)*(1+params[4]*T) 
