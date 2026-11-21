def equation(eps, T, params):
    return params[0]*eps+params[1]*np.sqrt(eps)/T+params[2]+params[3]*T+params[4]*T**2+params[5]/T+params[6]*np.log(T)+params[7]*T**3+params[8]/T**2+params[9]*T*np.log(T)