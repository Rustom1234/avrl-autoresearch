def equation(eps, T, params):
    L = np.log(eps)
    return np.exp(params[0] + params[1]*T + params[2]*L/T + params[3]*L**2 + params[4]*T*L + params[5]*T*L**2 + params[6]*L/T**2)
