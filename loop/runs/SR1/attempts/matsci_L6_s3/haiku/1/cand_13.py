def equation(eps, T, params):
    return -((params[0] + params[1]*T + params[2]/T)*(1 + params[3]*eps**params[4]/T + params[5]*eps/T) + params[6]*np.log(eps)/T)
