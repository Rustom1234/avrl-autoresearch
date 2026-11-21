def equation(eps, T, params):
    A = params[0] + params[1]*T + params[2]*T**2 + params[3]*T**3
    return A * eps**(params[4] + params[5]*T) * np.exp(-(params[6] + params[7]*T)*eps**params[8]) + params[9]*np.log(eps)
