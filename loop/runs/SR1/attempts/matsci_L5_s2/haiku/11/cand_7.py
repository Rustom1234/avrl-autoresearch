def equation(eps, T, params):
    return (params[0]+params[1]*T+params[2]*T**2)+ (params[3]+params[5]*T)*(eps+params[6])**params[4] + params[7]*eps
