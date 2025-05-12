import numpy as np
import math
from qutip import *
from qutip.measurement import measure, measurement_statistics
from scipy import interpolate
import seaborn as sns
from labellines import labelLines
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


# Plotting minor for TMSV with a coherent product state as the auxiliary:
# 
# $d'^\text{TMSV}_{1001}=\frac{1}{2}\left(d_{1001}^\text{TMSV}+\langle \boldsymbol{a}^\dagger\boldsymbol{b}^\dagger\rangle_\epsilon\langle \boldsymbol{a}\boldsymbol{b}\rangle_\epsilon\right)=\frac{1}{2}\left(\frac{-\lambda^2}{1-\lambda^2}+\left|\frac{\lambda}{1-\lambda^2}-\gamma\delta\right|^2\right)$


def tmsvaux(lambdaa,ab):
    return 0.5*(-(lambdaa**2)/(1-lambdaa**2)+abs((lambdaa/(1-lambdaa**2))-ab)**2)

tmsvauxvec=np.vectorize(tmsvaux)

y = np.arange(-0.9995, 1.0, 0.0005) #range of alpha*beta
x = np.arange(-0.9995, 1.0, 0.0005) #range of lambda\in(-1,1)
x, y = np.meshgrid(x, y)
Z = tmsvauxvec(x,y)
bottom=np.min(Z)*1.25 #defining lower limit of z-axis range
Z[Z>=0]=np.nan #clipping points geq 0


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, Z, color="slateblue",
                       linewidth=0, antialiased=False)
#CS = ax.contour(X, Y, Z, zdir='z', offset=-2, cmap='magma')
#ax.clabel(CS, [-1, 0.5,], inline=1, fontsize=10)
ax.set_zlim(bottom,0.0)
ax.set_ylabel(r'$\gamma\delta$',fontsize=14,labelpad=7.5)
ax.set_xlabel(r'$\lambda$',fontsize=14,labelpad=7.5)
ax.set_zlabel(r'$d^\prime_{1001}$',fontsize=14,labelpad=7.5)
ax.tick_params(labelsize=12)
plt.savefig('plots/tmsvwaux.pdf',dpi=300,bbox_inches='tight')
plt.show()


# Plotting minor for two-mode Schrödinger cat states with a coherent product state as the auxiliary:
# 
# $d'^\text{cat}_{mnpq}=\frac{1}{2}\left(d_{mnpq}^\text{cat}+\langle \boldsymbol{a}^{\dagger p}\boldsymbol{a}^m\boldsymbol{b}^{\dagger n}\boldsymbol{b}^q\rangle_\epsilon\langle \boldsymbol{a}^{\dagger m}\boldsymbol{a}^p\boldsymbol{b}^{\dagger q}\boldsymbol{b}^n\rangle_\epsilon\right)\\
# \quad\quad=\frac{1}{2}\left(|\alpha|^{2m+2n+2p+2q}+|\gamma|^{2m+2n+2p+2q}-2\text{Re}\left[(\alpha^*\gamma)^{p+n}(\alpha\gamma^*)^{m+q}(\frac{1+e^{-\Delta}}{1-e^{-\Delta}})\right]\right),$
# where $m\nsim q$ and $n\nsim p$, and $m$ or $p$, and $n$ or $q$, are zero.
# 
# The two-mode Schrödinger cat state is parametrised by $\alpha$, and the reference state by $\gamma$.
# 
# The cat state is odd: $\theta=\pi$.


#remember that m or p, and n or q, have to be zero
#taking alpha and beta to be real
def cataux(alpha,beta,p,m,n,q): 
    if alpha==0.0 or beta==0.0:
        return 0.0
    else:
        delta=2*(abs(alpha)**2+abs(alpha)**2)
        return 0.5*(abs(alpha)**(2*m+2*n+2*p+2*q)+abs(beta)**(2*m+2*n+2*p+2*q)-(2*(alpha*beta)**(m+n+p+q)*((1+np.exp(-delta))/(1-np.exp(-delta)))))

catauxvec=np.vectorize(cataux)

X = np.arange(0, 2, 0.005) #range of alpha
Y = np.arange(0, 2, 0.005) #range of beta
X, Y = np.meshgrid(X, Y)
Z1 = catauxvec(X,Y,0,1,1,0)
Z3 = catauxvec(X,Y,0,3,3,0)
Z5 = catauxvec(X,Y,0,5,5,0)
Z1[Z1>=0]=np.nan #clipping points geq 0
Z3[Z3>=0]=np.nan #clipping points geq 0
Z5[Z5>=0]=np.nan #clipping points geq 0



#separate individual plots

fig, ax = plt.subplots()
c1 = ax.pcolormesh(X, Y, Z1, cmap=cm.magma, vmax=0)
ax.set_title(r'$d^\prime_{1,1,0,0}$', pad=12, fontsize=22)
ax.set_xlabel(r'$|\alpha|$',fontsize=22)
ax.set_ylabel(r'$|\gamma|$',fontsize=22,rotation=0,labelpad=14)
ax.tick_params(labelsize=20)
ax.set_xticks([0,0.5,1.0,1.5,2.0])
ax.set_yticks([0,0.5,1.0,1.5,2.0])
cbar = plt.colorbar(c1)
cbar.ax.tick_params(labelsize=20)
plt.savefig('plots/cataux0110flat.png',dpi=300,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
c2 = ax.pcolormesh(X, Y, Z3, cmap=cm.magma, vmax=0)
ax.set_title(r'$d^\prime_{3,3,0,0}$', pad=12, fontsize=22)
ax.set_xlabel(r'$|\alpha|$',fontsize=22)
ax.set_ylabel(r'$|\gamma|$',fontsize=22,rotation=0,labelpad=14)
ax.tick_params(labelsize=20)
ax.set_xticks([0,0.5,1.0,1.5,2.0])
ax.set_yticks([0,0.5,1.0,1.5,2.0])
cbar = plt.colorbar(c2)
cbar.ax.tick_params(labelsize=20)
plt.savefig('plots/cataux0330flat.png',dpi=300,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
c3 = ax.pcolormesh(X, Y, Z5, cmap=cm.magma, vmax=0)
ax.set_title(r'$d^\prime_{5,5,0,0}$', pad=12, fontsize=22)
ax.set_xlabel(r'$|\alpha|$',fontsize=22)
ax.set_ylabel(r'$|\gamma|$',fontsize=22,rotation=0,labelpad=14)
ax.tick_params(labelsize=20)
ax.set_xticks([0,0.5,1.0,1.5,2.0])
ax.set_yticks([0,0.5,1.0,1.5,2.0])
cbar = plt.colorbar(c3)
cbar.ax.tick_params(labelsize=20)
plt.savefig('plots/cataux0550flat.png',dpi=300,bbox_inches='tight')
plt.show()



# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.45))

#first plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 1)
c1=ax.pcolormesh(X, Y, Z1, cmap=cm.magma, vmax=0)
ax.set_title(r'$d^\prime_{1100}$', fontsize=14, pad=10)
ax.set_ylabel(r'$|\gamma|$',fontsize=14,labelpad=10)
ax.set_xlabel(r'$|\alpha|$',fontsize=14)
ax.tick_params(labelsize=12)
ax.set_xticks([0,0.5,1.0,1.5,2.0])
ax.set_yticks([0,0.5,1.0,1.5,2.0])
cbar = plt.colorbar(c1,ax=ax,location='bottom',pad=0.15)
cbar.ax.tick_params(labelsize=12)

#second plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 2)
c2=ax.pcolormesh(X, Y, Z3, cmap=cm.magma, vmax=0)
ax.set_title(r'$d^\prime_{3300}$', fontsize=14, pad=10)
ax.set_xlabel(r'$|\alpha|$',fontsize=14)
ax.tick_params(labelsize=12)
ax.set_xticks([0,0.5,1.0,1.5,2.0])
ax.set_yticks([])
cbar = plt.colorbar(c2,ax=ax,location='bottom',pad=0.15)
cbar.ax.tick_params(labelsize=12)

#third plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 3)
c3=ax.pcolormesh(X, Y, Z5, cmap=cm.magma, vmax=0)
ax.set_title(r'$d^\prime_{5500}$', fontsize=14, pad=10)
ax.set_xlabel(r'$|\alpha|$',fontsize=14)
ax.tick_params(labelsize=12)
ax.set_xticks([0,0.5,1.0,1.5,2.0])
ax.set_yticks([])
cbar = plt.colorbar(c3,ax=ax,location='bottom',pad=0.15)
cbar.ax.tick_params(labelsize=12)

plt.savefig('plots/cataux.png',dpi=600,bbox_inches='tight')
plt.show()


# Plotting minor for N00N states with a coherent product state $|\gamma,\delta\rangle$ as the auxiliary:
# 
# $d'^{N00N}_{00NN}=\frac{1}{2}\left(d^{N00N}_{00NN}+\langle\boldsymbol{a}^{\dagger N}\boldsymbol{b}^N\rangle_\epsilon\langle\boldsymbol{a}^{N}\boldsymbol{b}^{\dagger N}\rangle_\epsilon\right)=\frac{1}{2}\left(-N!|\gamma\delta|^{2N}+|\gamma\delta|^{2N}\right)$


def n00nminor(N):
    ans=-0.25*(math.factorial(N))**2
    return ans

def n00naux(N,ab):
    ans=0.5*(-math.factorial(N)*(ab**N)+(ab)**(2*N))
    return ans

n00nminorvec=np.vectorize(n00nminor)
n00nauxvec=np.vectorize(n00naux)


fig, ax = plt.subplots()

N = np.arange(2, 7, 1) #range of N
X = np.arange(0, 3.5, 0.0005) #range of alpha*beta #0.0005

cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(1, 0, len(N)+1)]

for i in range(len(N)):
    n=int(N[i])
    plt.plot(X,n00nauxvec(n,X),color=colors[i+1],label=r'$\boldsymbol{ N= %i}$'%n)

lines = plt.gca().get_lines()
labelLines(lines, align=False,xvals=np.linspace(0.8,2.5,len(N)),fontsize=18,color='black')
ax.set_ylabel(r'$d^\prime_{00NN}$', fontsize=20)
ax.set_yscale('symlog',linthresh=0.00001)
ax.set_ylim([-10e4, -10e-3])
ax.set_xlim([0, 3])
ax.set_yticks([-10e4,-10e2,-10e0,-10e-2])
ax.set_xlabel(r'$\gamma\delta$',fontsize=20)
ax.tick_params(labelsize=18)
ax.xaxis.grid(which='major',ls='dashed',alpha=0.8)
ax.yaxis.grid(which='major',ls='dashed',alpha=0.8)
ax.yaxis.grid(which='minor',ls='dashed',alpha=0.8)
#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig('plots/n00naux.pdf',dpi=300,bbox_inches='tight')
plt.show()


# Lossy regime for N00N states (using two entangled states), with losses captured by $\eta$ and dephasing by $p$.


#NOTE: only valid in the case of zero dark counts!

#diagonal entries
def diag(m,n,l,k,dima,dimb,eta):
    return tensor((np.sqrt(eta)*create(dima))**m*(np.sqrt(eta)*destroy(dima))**n*(np.sqrt(eta)*create(dima))**n*(np.sqrt(eta)*destroy(dima))**m,(np.sqrt(eta)*create(dimb))**l*(np.sqrt(eta)*destroy(dimb))**k*(np.sqrt(eta)*create(dimb))**k*(np.sqrt(eta)*destroy(dimb))**l)

#off-diagonal entries
def offdiag(m,n,p,q,s,r,k,l,dima,dimb,eta):
    return tensor((np.sqrt(eta)*create(dima))**m*(np.sqrt(eta)*destroy(dima))**n*(np.sqrt(eta)*create(dima))**p*(np.sqrt(eta)*destroy(dima))**q,(np.sqrt(eta)*create(dimb))**s*(np.sqrt(eta)*destroy(dimb))**r*(np.sqrt(eta)*create(dimb))**k*(np.sqrt(eta)*destroy(dimb))**l)

#the minor witnesses entanglement only in the case of m=l=N and the rest 0, or any equivalent permutation
def n00nminor(m,n,p,q,s,r,k,l,N,alpha,beta,parameter,eta): 
    #N = number of photons in \ket{N}
    #eta = detector efficiency
    d=N+1 #+1 needed as N=dim(H)-1
    dima=d+max(n,p)
    dimb=d+max(r,k) 
    #dephased NOON state parametrised by p\in[0,1]; p=1 gives separable state, fully dephased
    rho=parameter*(alpha*tensor(basis(dima,N),basis(dimb,0))*tensor(basis(dima,N),basis(dimb,0)).dag()*np.conj(alpha)+beta*tensor(basis(dima,0),basis(dimb,N))*tensor(basis(dima,0),basis(dimb,N)).dag()*np.conj(beta))+(1-parameter)*(alpha*tensor(basis(dima,N),basis(dimb,0))*tensor(basis(dima,0),basis(dimb,N)).dag()*np.conj(beta)+beta*tensor(basis(dima,0),basis(dimb,N))*tensor(basis(dima,N),basis(dimb,0)).dag()*np.conj(alpha))
    #defining the elements of the sub-matrix
    a00=(diag(m,n,l,k,dima,dimb,eta)*rho).tr()
    a11=(diag(q,p,s,r,dima,dimb,eta)*rho).tr()
    a01=(offdiag(m,n,p,q,s,r,k,l,dima,dimb,eta)*rho).tr()
    return (a00*a11-a01*np.conj(a01)).real #capturing real part only to avoid warning of ignoring imaginary part

n00nminorvec=np.vectorize(n00nminor)


# Lossy regime for cat states (using two entangled states), with losses captured by $\eta$ and dephasing by $p$.

#diagonal entries
def amn(alpha,beta,phi,m,n,pp,eta): #\braket{n_a^m n_b^n}; pp is dephasing parameter
    #returns division by zero if alpha=beta=0 and phi=np.pi
    if alpha !=0 and beta !=0:
        if (m+n) % 2 ==0: #m and n are of same parity
            return abs(np.sqrt(eta)*alpha)**(2*m)*abs(np.sqrt(eta)*beta)**(2*n)
        else: #(m,q) and (p,n) are of different parity
            return abs(np.sqrt(eta)*alpha)**(2*m)*abs(np.sqrt(eta)*beta)**(2*n)*(1-(1-pp)*np.cos(phi)*np.exp(-2*abs(alpha)**2-2*abs(beta)**2))/(1+(1-pp)*np.cos(phi)*np.exp(-2*abs(alpha)**2-2*abs(beta)**2))
    else:
        return 0.0

#off-diagonal entries
def apmnq(alpha,beta,phi,p,m,n,q,pp,eta): #power-ordering same as in notes; pp is dephasing parameter
    #returns division by zero if alpha=beta=0 and phi=np.pi
    if alpha !=0 and beta !=0:
        if (m+q) % 2 == 0 and (p+n) % 2 == 0: #(m,q) and (p,n) are of same parity
            return abs(np.sqrt(eta)*alpha)**(2*p+2*m)*abs(np.sqrt(eta)*beta)**(2*n+2*q)
        elif (m+q) % 2 != 0 and (p+n) % 2 != 0: #(m,q) and (p,n) are of different parity
            return abs(np.sqrt(eta)*alpha)**(2*p+2*m)*abs(np.sqrt(eta)*beta)**(2*n+2*q)*(1-(1-pp)*np.cos(phi)*np.exp(-2*abs(alpha)**2-2*abs(beta)**2))**2/(1+(1-pp)*np.cos(phi)*np.exp(-2*abs(alpha)**2-2*abs(beta)**2))**2
        else: #otherwise
            return abs(np.sqrt(eta)*alpha)**(2*p+2*m)*abs(np.sqrt(eta)*beta)**(2*n+2*q)*(1-p)**2*np.sin(phi)**2*np.exp(-4*abs(alpha)**2-4*abs(beta)**2)/(1+(1-p)*np.cos(phi)*np.exp(-2*abs(alpha)**2-2*abs(beta)**2))**2
    else:
        return 0.0

def catminor(alpha,beta,phi,p,m,n,q,pp,eta):
    if alpha !=0 and beta !=0: #checking that coherent state amplitudes are non-vanishing; otherwise return 0
        a00=amn(alpha,beta,phi,p,q,pp,eta)
        a01a10=apmnq(alpha,beta,phi,p,m,n,q,pp,eta)
        a11=amn(alpha,beta,phi,m,n,pp,eta)
        return (a00*a11-a01a10).real #.real to ensure no warnings when plotting, the value is real by definition
    else:
        return 0.0
    
catminorvec=np.vectorize(catminor)



N=2
X1 = np.arange(0, 1, 0.01) #range of dephasing p\in[0,1]
Y1 = np.arange(0, 1, 0.01) #range of efficiency parameter eta
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = n00nminorvec(N,0,0,0,0,0,0,N,N,np.sqrt(0.5),np.sqrt(0.5),Y1,X1)
bottom1 = np.min(Z1)*1.25 #defining lower limit of z-axis range

eta2 = np.arange(0.0, 1.01, 0.005) #range of eta\in[0,1]
alpha2 = np.arange(0.01, 3.001, 0.005) #range of alpha
eta2, alpha2 = np.meshgrid(eta2, alpha2)
Z2 = catminorvec(alpha2,alpha2,np.pi,0,1,1,0,0,eta2)
bottom2=np.min(Z2)*1.25 #defining lower limit of z-axis range

p3 = np.arange(0, 1.01, 0.005) #range of dephasing parameter
alpha3= np.arange(0.01, 3.001, 0.005) #range of alpha
alpha3, p3 = np.meshgrid(alpha3, p3)
Z3 = catminorvec(alpha3,alpha3,np.pi,0,1,1,0,p3,1)
bottom3=np.min(Z3)*1.25 #defining lower limit of z-axis range



# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.333))

#first plot
# set up the Axes for the first plot
ax = fig.add_subplot(1, 3, 1, projection='3d')
Z1[Z1>=0]=np.nan #clipping points geq 0
surf = ax.plot_surface(X1, Y1, Z1, cmap=cm.magma,
                       linewidth=0, antialiased=False)
ax.set_zlim(bottom1,0.0)
ax.set_ylabel(r'$p$',fontsize=16,labelpad=10)
ax.set_xlabel(r'$\eta$',fontsize=16,labelpad=8)
ax.set_zlabel(r'$d^\prime_{0022}$',fontsize=16,labelpad=8)
ax.set_title(r'$\textrm{(a)}$',fontsize=16)
ax.tick_params(labelsize=14)

# second plot
ax = fig.add_subplot(1, 3, 2, projection='3d')
Z2[Z2>=0]=np.nan #clipping points geq 0
surf = ax.plot_surface(eta2, alpha2, Z2, cmap=cm.magma,
                       linewidth=0, antialiased=False)
ax.set_zlim(bottom2,0.0)
ax.set_xlabel(r'$\eta$',fontsize=16,labelpad=8)
ax.set_ylabel(r'$\alpha$',fontsize=16,labelpad=10)
ax.set_zlabel(r'$d^\prime_{1100}$',fontsize=16,labelpad=8)
ax.set_title(r'$\textrm{(b)}$',fontsize=16)
ax.tick_params(labelsize=14)

#third plot
ax = fig.add_subplot(1, 3, 3, projection='3d')
Z3[Z3>=0]=np.nan #clipping points geq 0
surf = ax.plot_surface(alpha3, p3, Z3, cmap=cm.magma,
                       linewidth=0, antialiased=False)
ax.set_zlim(bottom3,0.0)
ax.set_xlabel(r'$\alpha$', fontsize=16, labelpad=8)
ax.set_ylabel(r'$p$', fontsize=16, labelpad=10)
ax.set_zlabel(r'$d^\prime_{1100}$',fontsize=16,labelpad=8)
ax.set_title(r'$\textrm{(c)}$',fontsize=16)
ax.tick_params(labelsize=14)

plt.savefig('plots/lossy.pdf',dpi=300,bbox_inches='tight')
plt.show()


# In the case of $N00N$ states with identical copies, we may apply Hoeffding's inequality to obtain the critical number of measurements $m_0$ for $\epsilon$-accuracy with confidence of $(1-\delta)$:
# 
# $m_0=\frac{(2N+1)^2 N^{4N}}{2\epsilon^2}\ln\frac{2}{\delta}$.


#tensor product of two identical N00N states, taking trace over modes c2 and d2
def N00N2(N,phi,pphi): 
    #truncating the Fock space past number state d
    d=2*N+1 #there can be at most 2N photons ending up on a single mode
    op1=(tensor(create(d)*np.exp(1j*phi),qeye(d),qeye(d),qeye(d))+tensor(qeye(d),create(d)*np.exp(1j*phi),qeye(d),qeye(d)))**N
    op2=(tensor(qeye(d),qeye(d),create(d)*np.exp(1j*pphi),qeye(d))+tensor(qeye(d),qeye(d),qeye(d),create(d)*np.exp(1j*pphi)))**N
    op3=(tensor(create(d),qeye(d),qeye(d),qeye(d))-tensor(qeye(d),create(d),qeye(d),qeye(d)))**N
    op4=(tensor(qeye(d),qeye(d),create(d),qeye(d))-tensor(qeye(d),qeye(d),qeye(d),create(d)))**N
    state2=(0.5*0.5**(N)*(op1+op2)*(op3+op4)*tensor(basis(d,0),basis(d,0),basis(d,0),basis(d,0))).unit()
    state2 = state2.ptrace([0,2])
    return state2

#photon-number operator, all to the power of N
def cdagcddagd(N): 
    d=2*N+1 #there can be at most 2N photons ending up on a single mode
    #modes (c1,d1)
    measurement = tensor((create(d)*destroy(d))**N,(create(d)*destroy(d))**N)
    return measurement

#finding the standard deviation of the number distribution for given phi, phi'
def std(N,phi,pphi):
    #x,y,z = eigenvalues, eigenvectors, probabilities
    x,y,z = measurement_statistics(N00N2(N,phi,pphi), cdagcddagd(N))
    exp=(x*np.array(z)).sum()/np.array(z).sum()
    exp2=(x**2*np.array(z)).sum()/np.array(z).sum()
    return np.sqrt(exp2-(exp)**2)

def m0n00n(N,error,delta): # epsilon = error * sigma
    d=2*N+1
    m0=0
    i=0
    while i<d:
        j=0
        while j<d:
            epsilon=std(N,2*np.pi*i/d,2*np.pi*j/d) #error set to 1sigma for the given distribution at phi,pphi
            epsilon=error*epsilon
            m0=m0+(np.log(2/delta)*N**(4*N)/(2*epsilon**2))
            j=j+1
        i=i+1
    return m0

# m_0 for negative error margins
def m0n00nneg(N,delta): # epsilon = \pm|d_{00NN}|
    epsilon=abs(n00nminor(N))
    m0=(2*N+1)**(2)*N**(4*N)*np.log(2/delta)/(2*epsilon**2)
    
    return m0
            
m0n00nvec=np.vectorize(m0n00n)
m0n00nnegvec=np.vectorize(m0n00nneg)



N=np.arange(1,9,1)
m01=m0n00nvec(N,0.5,0.1)
m02=m0n00nvec(N,1.0,0.1)
yn00n=n00nminorvec(N)


fig, ax1 = plt.subplots()

ax1.plot(N,m01,lw=0.75,label=r'$\epsilon=0.5 \sigma,\,\delta=0.1$',marker='D',color='purple')   
ax1.plot(N,m02,lw=0.75,label=r'$\epsilon=1\sigma,\,\delta=0.1$',marker='D',color='orange')
plt.yscale('log')
ax1.set_xlabel(r'$N$',fontsize=22)
ax1.set_ylabel(r'$m_0$',fontsize=22,rotation=0,labelpad=14)
ax1.tick_params(labelsize=20)
ax1.set_xticks(N)
ax1.set_yticks([1e+2,1e+4,1e+6,1e+8])
legend = ax1.legend(fancybox=False, edgecolor="black",fontsize=20)
legend.get_frame().set_linewidth(0.5)
ax1.set_title(r'$(b)$', pad=12, fontsize=22)
plt.gca().set_aspect('equal')
plt.savefig('plots/m0n00n.pdf',dpi=300,bbox_inches='tight')
plt.show()


# Extending the analysis to two-mode Schr\"{o}dinger cat states, $N(\alpha)\left[|\alpha,\alpha\rangle-|-\alpha,-\alpha\rangle\right]$. Here, we have to make use of the Chebyshev inequality since the eigenspectrum is unbounded and the distribution is not sub-Gaussian. The minimum number of measurements is then determined by
# 
# $m_0 = \frac{\textrm{Var}\left(\hat{A}\right)}{\epsilon^2 \delta}$,
# 
# where $\hat{A}$ is the photon-number operator for which we are estimating the expectation value. This also needs an additional factor of $(2p+1)^2$, where $p$ is the largest frequency component, in order to account for the Fourier analysis.


# taking coherent state using Fock state representation and truncating past n=20
# for alpha = 2, prob(n=20) = 8e-15
amp=20

# defining the input cat state
def catstate(alpha):
    a1=tensor(displace(amp,alpha)*basis(amp,0),displace(amp,alpha)*basis(amp,0))
    a2=tensor(displace(amp,-alpha)*basis(amp,0),displace(amp,-alpha)*basis(amp,0))
    state=(a1-a2).unit()
    return state

# defining the cat state post-beam splitter
def catstate2(alpha,phi,pphi):
    a1=tensor(displace(amp,(np.exp(1j*phi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(np.exp(1j*pphi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(np.exp(1j*phi)-1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(np.exp(1j*pphi)-1)*alpha*np.sqrt(0.5))*basis(amp,0))
    a2=tensor(displace(amp,-(np.exp(1j*phi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,-(np.exp(1j*pphi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,-(np.exp(1j*phi)-1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,-(np.exp(1j*pphi)-1)*alpha*np.sqrt(0.5))*basis(amp,0))
    a3=tensor(displace(amp,(np.exp(1j*phi)-1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(np.exp(1j*pphi)-1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(np.exp(1j*phi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(np.exp(1j*pphi)+1)*alpha*np.sqrt(0.5))*basis(amp,0))
    a4=tensor(displace(amp,(-np.exp(1j*phi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(-np.exp(1j*pphi)+1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(-np.exp(1j*phi)-1)*alpha*np.sqrt(0.5))*basis(amp,0),displace(amp,(-np.exp(1j*pphi)-1)*alpha*np.sqrt(0.5))*basis(amp,0))
    return (a1+a2-a3-a4).ptrace([0,2])

# operator a^dagger m a^m b^dagger n b^n
def abcat(m,n): 
    measurement = tensor((create(amp)**m*destroy(amp)**m),(create(amp)**n*destroy(amp)**n))
    return measurement

# variance for the photon-number operator abcat(m,n)
def varcat1(alpha,m,n):
    #x,y,z = eigenvalues, eigenvectors, probabilities
    x,y,z = measurement_statistics(catstate(alpha), abcat(m,n))
    exp=(x*np.array(z)).sum()/np.array(z).sum()
    exp2=(x**2*np.array(z)).sum()/np.array(z).sum()
    return exp2-(exp)**2

# photon-number operator of order p+q
def cdcat(p,q): 
    #modes (c1,d1)
    measurement = tensor((create(amp)*destroy(amp))**p,(create(amp)*destroy(amp))**q)
    return measurement

# variance for the photon-number operator cdcat(p,q)
def varcat2(alpha,p,q,phi,pphi):
    #x,y,z = eigenvalues, eigenvectors, probabilities
    x,y,z = measurement_statistics(catstate2(alpha,phi,pphi), cdcat(p,q))
    exp=(x*np.array(z)).sum()/np.array(z).sum()
    exp2=(x**2*np.array(z)).sum()/np.array(z).sum()
    return exp2-(exp)**2

# WARNING: only works for d_{m,n,p,q} with either m,n = 0 or p,q = 0
# var(X+Y) = var(X) + var (Y) when X and Y are independent random variables
def totalvar(alpha,m,n,p,q):
    variance,i=0,0
    while i<(2*p+1):
        j=0
        while j<(2*q+1):
            variance=variance+varcat2(alpha,p,q,2*np.pi*i/(2*p+1),2*np.pi*j/(2*q+1))
            j=j+1
        i=i+1
    ans=variance+varcat1(alpha,m,n)
    return ans

# WARNING: only works for d_{m,n,p,q} with either m,n = 0 or p,q = 0
def catminor(alpha,m,n,p,q):
    delta=4*abs(alpha)**2
    def a1(m,n):
        if (m+n)%2==0:
            return abs(alpha)**(2*m+2*n)
        else:
            return abs(alpha)**(2*m+2*n)*(1+np.exp(-delta))/(1-np.exp(-delta))
    def a2(m,n,p,q):
        if (m+q)%2==0 and (n+p)%2==0:
            return abs(alpha)**(2*(p+n+m+q))
        if (m+q)%2==1 and (n+p)%2==1:
            return abs(alpha)**(2*(p+n+m+q))*((1+np.exp(-delta))/(1-np.exp(-delta)))**2
        else:
            return 0
    return a1(m,n)-a2(m,n,p,q)

totalvarvec=np.vectorize(totalvar)
catminorvec=np.vectorize(catminor)



delta=0.1
x=np.arange(0.01,1.6,0.025)
y=catminorvec(x,1,1,0,0)
#yerr=0.001*totalvarvec(x,5,5,0,0)
yerr=np.linspace(0,0,len(x))
for i in range(len(x)):
    yerr[i]=min(0.025,abs(catminor(x[i],1,1,0,0)))
    i=i+1
#yerr=np.linspace(0.1,0.1,len(x))
m0=totalvarvec(x,1,1,0,0)/(delta*(yerr)**2)



fig, ax2 = plt.subplots()

color = 'purple'
ax2.plot(x, np.ceil(m0), color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xlabel(r'$\alpha$',fontsize=22)
ax2.set_ylabel(r'$m_0$',color=color,fontsize=22,labelpad=0)
ax2.tick_params(labelsize=20)
plt.yscale('log')
plt.xlim(0,1.5)
ax2.set_title(r'$(a)$', pad=12, fontsize=22)

ax3 = ax2.twinx()
color = 'brown'
ax3.set_ylabel(r'$d_{1,1,0,0}$',fontsize=22,labelpad=0, color=color)  # we already handled the x-label with ax1
ax3.plot(x,np.linspace(0,0,len(x)),lw=0.75,color=color,linestyle='dashed')
ax3.plot(x, y,lw=0.75,label=r'$\epsilon=0.5 \sigma,\,\delta=0.1$',color='saddlebrown')   
ax3.fill_between(x, y-yerr, y+yerr, alpha=0.2, facecolor='orange')
ax3.tick_params(axis='y', labelcolor=color,labelsize=20)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig('plots/m0catd1100.pdf',dpi=300,bbox_inches='tight')
plt.show()


delta=0.1
x=np.arange(0.01,1.6,0.1)
y=catminorvec(x,3,3,0,0)
#yerr=0.001*totalvarvec(x,5,5,0,0)
yerr=np.linspace(0,0,len(x))
for i in range(len(x)):
    yerr[i]=min(0.25,abs(catminor(x[i],3,3,0,0)))
    i=i+1
#yerr=np.linspace(0.1,0.1,len(x))
m0=totalvarvec(x,3,3,0,0)/(delta*(yerr)**2)


cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 2, 3)]
color1 = '#470C6D'
color2 = '#BE7600'
color3 = '#519A76'

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.45))

# first plot
ax1 = fig.add_subplot(1, 2, 1)

ax1.tick_params(axis='y', labelcolor=color1)
ax1.plot(x, np.ceil(m0), color=color1)
ax1.set_xlabel(r'$\alpha$',fontsize=22)
ax1.set_ylabel(r'$m_0$',color=color1,fontsize=22,labelpad=5)
ax1.tick_params(labelsize=20)
ax1.set_axisbelow(True)
ax1.xaxis.grid(ls='dashed',alpha=0.75)
ax1.yaxis.grid(which='major',ls='dashed',alpha=0.75)
ax1.yaxis.grid(which='minor',ls='dashed',alpha=0.75,color=color1)
plt.yscale('log')
plt.xlim(0,1.5)
plt.ylim(1,1e9)
ax1.set_yticks([1e1,1e3,1e5,1e7])
ax1.set_title(r'$\mathrm{(a)}$', pad=12, fontsize=22)

ax3 = ax1.twinx()
ax3.set_ylabel(r'$d_{1100}$',fontsize=22,labelpad=0, color=color2)  # we already handled the x-label with ax1
ax3.plot(x,np.linspace(0,0,len(x)),color=color2,lw=0.95,linestyle='dashed')
ax3.plot(x, y,color=color2)   
ax3.fill_between(x, y-yerr, y+yerr,label=r'$d_{1100}\pm\epsilon$', alpha=0.2, facecolor='orange')
ax3.tick_params(axis='y', labelcolor=color2,labelsize=20)
legend = ax3.legend(fancybox=False, facecolor='white',edgecolor="black",fontsize=18,labelcolor=color2)
legend.get_frame().set_linewidth(0.5)
ax3.set_xlabel(r'$\alpha$',fontsize=22)

# second plot
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(N,m01,label=r'$\epsilon=0.5 \sigma$',marker='o',color=color1)   
ax2.plot(N,m02,label=r'$\epsilon=1\sigma$',marker='d',color=color3)
plt.yscale('log')
ax2.set_xlabel(r'$N$',fontsize=22)
ax2.set_ylabel(r'$m_0$',fontsize=22,rotation=90,labelpad=5)
ax2.tick_params(labelsize=20)
ax2.set_xticks(N)
plt.xlim(np.min(N),np.max(N))
ax2.set_yticks([1e1,1e3,1e5,1e7])
#ax2.set_yticklabels([])
plt.ylim(1,1e9)
legend = ax2.legend(fancybox=False, edgecolor="black",fontsize=18)
legend.get_frame().set_linewidth(0.5)
ax2.yaxis.grid(which='major',ls='dashed',alpha=0.75)
ax2.yaxis.grid(which='minor',ls='dashed',alpha=0.75)
ax2.set_title(r'$\mathrm{(b)}$', pad=12, fontsize=22)

fig.tight_layout() # otherwise right y-labels are slightly clipped
plt.savefig('plots/m0.pdf',dpi=300,bbox_inches='tight')
plt.show()




