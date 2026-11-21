def equation(eps,T,params):
    return params[0]/(eps**2+params[1])+params[2]+params[3]*np.log(eps)+params[4]*eps+params[5]*T**2*eps**3+params[6]*eps**2+params[7]*eps**3