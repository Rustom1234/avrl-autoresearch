def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*T**2 + params[3]*T**3 + params[4]*T**4 + params[5]*np.sqrt(T)*eps + params[6]*np.log(T)