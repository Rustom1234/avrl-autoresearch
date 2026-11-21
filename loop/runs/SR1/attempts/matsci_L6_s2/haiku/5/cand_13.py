def equation(eps, T, params):
    return params[0]*np.exp(params[1]*T)*eps**(params[2]/T)*np.exp(params[3]*eps**2/T + params[4]*np.log(eps)**2)
