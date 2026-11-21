def equation(eps, T, params):
    return -(params[0] + params[1]*T)*(eps+params[2])**(params[3] + params[4]*T)
