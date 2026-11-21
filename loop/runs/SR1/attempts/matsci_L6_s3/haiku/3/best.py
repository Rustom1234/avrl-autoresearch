def equation(eps, T, params):
    return params[0] + params[1]*np.log(T) + params[2]/T + params[3]*np.sqrt(eps)/T + params[4]*eps + params[5]*eps/T + params[6]*T
