def equation(eps, T, params):
    return -(params[0] + params[1]*np.log(T) + params[2]*eps + params[3]*eps*T + params[4]*eps**2 + params[5]*eps*np.log(T) + params[6]*eps*np.exp(-params[7]*eps)*T + params[8]*eps**2*T)
