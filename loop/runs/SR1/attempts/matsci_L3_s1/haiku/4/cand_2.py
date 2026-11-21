def equation(eps, T, params):
    return (params[0] + params[1]*T) * (1-np.exp(-params[2]*eps)) * np.exp(-params[3]*eps) * (1 + params[4]*T) + params[5]*eps*0
