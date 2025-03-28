import math as m

def problem1(mut,muc,dc,dmaj,p,F):
    dm = dmaj-(p/2)
    term1 = ((F*dm)/2)
    term3 = (F*muc*dc)/2
    term2R = ((3.14159*mut*dm)+p)/(3.14159*dm-(mut*p))
    term2L = ((3.14159*mut*dm)-p)/(3.14159*dm+(mut*p))
    Tr = (term1*term2R)+term3
    Tl = (term1*term2L)+term3
    n = (F*p)/(2*3.1415*Tr)
    print('Tr:',Tr)
    print('Tl:',Tl)
    print('efficiency:',n)
    

def problem2(Sy,L,d):
    Force = (Sy*3.14159*(d**3))/(32*(L-d))
    print('F:',Force)


#a = alpha = 29
def problem3(thread,d,a,F,mu,muc,l,fd):
    p = 1/thread
    dm = d-(p/2)
    print(dm)
    theta = a/2
    t = theta*(3.1415/180)
    sec = 1/(m.cos(t))
    term1 = (dm)/2
    term2 = (p+(sec*dm*3.14159*mu))/((3.14159*dm)-(mu*p*sec))
    term3 = (muc*fd)/2
    Ttot = (term1*term2)+term3
    print(term2)
    Ttot2 = F*l
    ans = Ttot2/Ttot
    print("F:",ans)


problem3(6,.375,29,67.92,.15,.15,3.5,1)

