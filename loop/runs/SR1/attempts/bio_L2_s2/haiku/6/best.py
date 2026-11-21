def equation(t,P,params):
    a,b,c,d,e,f,g,h,k,m=params
    return P*(a+b*np.exp(-c*t)) + P**2*(d*np.exp(-e*t)+f) + g*P**3 + h*np.exp(-k*t) + m