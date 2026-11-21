def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*np.log(eps) + params[3]*T*np.log(eps) + params[4]*eps*T + params[5]*eps + params[6]*np.log(T)