def equation(eps, T, params):
    return params[0]+params[1]*T**params[2]+params[3]*eps**params[4]*T**params[5]+params[6]*np.log(eps)*T**params[7]+params[8]*eps**(-np.abs(params[9]))/np.sqrt(T)