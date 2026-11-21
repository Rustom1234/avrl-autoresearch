def equation(eps, T, params):
    return (params[0]/T + params[1]*T**params[4]) * eps*np.exp(-params[2]*eps) + params[3] + params[5]*T
