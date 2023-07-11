#QRM TPC rates
#Priya Nadkarni and Praveen Jayakumar

import math
import matplotlib.pyplot as plt

def binom_sum(m,start,end):
    sum_binom = 0
    for i in range(end-start+1):
        sum_binom += math.comb(m,start+i)
    return sum_binom
 
def catalytic_rate(r, m): #TPC
    m1=m
    m2=m # m2 > m1
    r1 = r # From RM(0,m1) code
    r2 = r# m2-r-1 # r2 = 0
    return (pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)))/pow(2, m1+m2)

points = []
rs = []
ms = []
m_list = list(range(1, 100))
for m in m_list:
    for r in range(int((m-1)/2)+1):
        if m-r-1 < m/2: #checking not self-dual containing.
            continue
        cr = catalytic_rate(r, m)
        if cr > 0:
            #print(r, m, cr)
            points.append([r, m])
            rs.append(r)
            ms.append(m)

#adding theorem 3 bound
def lr(r):
    diff = 1
    i = 0
    prods = [math.comb(2*r, u) for u in range(r+1)]
    while diff > 0:
        i+=1
        prods = [p*(2*r + i)/(2*r + i -u) for u, p in enumerate(prods)]
        diff = sum(prods) - ((2**(2*r + i))/(2 + math.sqrt(2)))
    return i-1

rs_thm = []
ms_thm = []
points_thm = []

for r in range(max(rs)+1):
    l =  lr(r)
    m_allowed = [2*r + s for s in range(1, l+1)]
    for m in m_allowed:
        if m-r-1 < m/2: #checking not self-dual containing.
            continue
        cr = catalytic_rate(r, m) #sanity check
        if cr > 0:
            #print(r, m, cr)
            points_thm.append([r, m])
            rs_thm.append(r)
            ms_thm.append(m)

r_lemma = list(range(max(rs)+1))
m_lemma = [r*2 + 2 for r in r_lemma]

r_lemma_n = list(range(max(rs)+1))
m_lemma_n = [r*3 + 1 for r in r_lemma]

plt.scatter(rs_thm, ms_thm, s = 15, label = 'Theorem 3')
plt.scatter(rs, ms, s=2, label = 'found by brute force')
plt.plot(r_lemma, m_lemma, label = 'lemma 6, m = 2r+2', color='r')
#plt.plot(r_lemma_n, m_lemma_n, label = 'm = 3r+1', color='g')
plt.legend()
plt.xlabel('r')
plt.ylabel('m')
plt.savefig('EATPRM.png')
 
def EA_rate(r, m):
    m1 =m
    m2 =m # m2 > m1
    r1 =r # From RM(0,m1) code
    r2 =r# m2-r-1 # r2 = 0
    return (pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)) + (binom_sum(m1,r1+1,m1-r1-1)*binom_sum(m2,r2+1,m2-r2-1)))/pow(2, m1+m2)

def catalytic_rate(r, m):
    m1=m
    m2=m # m2 > m1
    r1 = r # From RM(0,m1) code
    r2 = r# m2-r-1 # r2 = 0
    print("EA Rate = ", (pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)) + (binom_sum(m1,r1+1,m1-r1-1)*binom_sum(m2,r2+1,m2-r2-1)))/pow(2, m1+m2))
    print("Catalytic rate = ", (pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)))/pow(2, m1+m2))
    print("1(b) Rate in last step = ", binom_sum(m1,0,r1)*(binom_sum(m2,m2-r2,m2) + binom_sum(m2,0,r2)))
    m1=4
    m2=10 # m2 > m1
    r1 = 0 # From RM(0,m1) code
    r=8 # r > (m1+m2)/2 = 7 and r > (m2-1)/2 = 4.5
    r2 = m2-r-1 # r2 = 1
    print("2(a) Rate in first step = ", pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)) + (binom_sum(m1,r1+1,m1-r1-1)*binom_sum(m2,r2+1,m2-r2-1)))
    print("2(b) Rate in last step = ", binom_sum(m1,0,r1)*(binom_sum(m2,m2-r2,m2) + binom_sum(m2,0,r2)))
 
def EA_rate(r1, r2, m1, m2):
    return (pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)) + (binom_sum(m1,r1+1,m1-r1-1)*binom_sum(m2,r2+1,m2-r2-1)))/pow(2, m1+m2)

def catalytic_rate_assym(r1, r2, m1, m2):
    return (pow(2, m1+m2) - (2*binom_sum(m1,0,m1-r1-1)*binom_sum(m2,0,m2-r2-1)))/pow(2, m1+m2)

def catalytic_rate_RM(r, m):
    return (2*(binom_sum(m, 0, r)) - pow(2, m))/pow(2, m)

def obtain_l_r(r):
    for m0 in range(300):
        m = 2*r+1 + m0
        if binom_sum(m,0,r) <= (pow(2,m)/(2+np.sqrt(2))):
            return m-1

def main(l, u):
    for r in range(l, u):
        print("r = ", r+1, ", lower bound = ", (2 * (r+1) + 2),", l(r) = ", obtain_l_r(r+1), ", difference = ", obtain_l_r(r+1) -(2 * (r+1) + 2)+1)

main(530, 540)