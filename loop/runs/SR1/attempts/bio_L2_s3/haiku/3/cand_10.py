def equation(t, P, params):
    a,b,c,d,e,K = params[0],params[1],params[2],params[3],params[4],params[5]
    return (a+b*t)*P*np.log(K/P) + c + d*t + e*t*P + params[6]*P**2