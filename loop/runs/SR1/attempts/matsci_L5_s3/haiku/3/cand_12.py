def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]*T**2 + params[3]*np.log(eps) + params[4]*eps + params[5]*np.sqrt(eps) + params[6]*eps**2