def equation(eps, T, params):
    s = np.sqrt(T)
    return params[0]*s*eps + params[1] + params[2]*s + params[3]*T*T + params[4]*T*np.exp(-params[5]/T) + params[6]*T