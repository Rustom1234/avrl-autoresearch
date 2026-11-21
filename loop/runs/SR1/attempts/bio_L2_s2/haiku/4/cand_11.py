def equation(t, P, params):
    s=np.sin(params[4]*t+params[5]); 
    return P*(params[0]+params[1]*s+params[6]*t) - params[2]*P**2 + params[3]*s + params[7]*t + params[8]