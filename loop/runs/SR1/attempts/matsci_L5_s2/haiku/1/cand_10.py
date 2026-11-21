def equation(eps, T, params):
    return (params[0]*eps**params[1] + params[2]*eps**params[3])*(1+params[4]*T) + params[5]*T*eps**params[6] + params[7]*np.log(1+params[8]*eps)
