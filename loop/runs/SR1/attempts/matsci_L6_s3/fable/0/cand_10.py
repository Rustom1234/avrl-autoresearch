def equation(eps, T, params):
    return params[0]+params[1]*T**params[2]+params[3]*eps+params[4]*np.abs(eps)**params[5]*T**params[6]+params[7]*np.log(T)