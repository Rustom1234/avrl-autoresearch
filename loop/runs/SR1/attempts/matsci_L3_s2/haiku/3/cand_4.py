def equation(eps, T, params):
    a=params[0]+params[1]*np.exp(-params[2]*T)
    return a + (params[3]+params[4]*np.exp(-params[5]*T))*np.exp(-params[6]*eps) + params[7]*np.log(eps)*np.exp(-params[8]*eps) + params[9]*eps**2*0
