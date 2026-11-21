def equation(eps, T, params):
    return params[0]*eps + params[1]*eps**2*T**2 + params[2] + params[3]*np.log(T+params[4])