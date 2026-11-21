def equation(eps, T, params):
    return (params[0]/T + params[1] + params[2]*T) * eps*np.exp(-params[3]*eps) + params[4] + params[5]/T + params[6]*T**2
