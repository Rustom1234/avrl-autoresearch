def equation(t,P,params):
    A,k,a,b,c,d=params[0],params[1],params[2],params[3],params[4],params[5]
    T=np.tanh(k*t)
    return -A*T + a*P + b*P**2 + c*P**3 + d*P*T
