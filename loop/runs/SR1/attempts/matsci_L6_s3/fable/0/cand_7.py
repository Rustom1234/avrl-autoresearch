def equation(eps, T, params):
    return params[0]+params[1]*np.exp(-params[2]*T)+params[3]*np.abs(eps)**params[4]*np.exp(-params[5]*T)+params[6]*eps*np.exp(-params[7]*T)+params[8]*eps+params[9]*T