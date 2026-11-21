def equation(eps, T, params):
    return (params[0]+params[1]*np.log(1+params[2]*eps)+params[8]*eps)*(1-params[3]*T**params[4]) + params[5]*T**params[6]+params[7]