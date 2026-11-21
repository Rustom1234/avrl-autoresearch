def equation(t,P,params):
    A,k,b,c,d=params[0],params[1],params[2],params[3],params[4]
    return -A*np.tanh(k*t)*(1+b*P+c*P**2+d*P**3)
