def equation(t, x, v, params):
    A,w,ph,c,d,k,b,g,e,q=params
    return (A+e*v)*np.cos(w*t+ph) - c*v - d*v*np.abs(v) - k*x - b*x**3 + g + q*x*v
