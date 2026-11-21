def equation(eps, T, params):
    return params[0] + params[1]*np.log(1+eps) + params[2]*eps + params[3]*eps**2 + params[4]*T*eps + params[5]*T*np.log(1+eps)
