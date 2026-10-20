import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

#1

X = np.linspace(0,2*np.pi,1000)

#2
def sin_der(x) : 
    grad_v = np.vectorize(grad(np.sin))
    return grad_v(x)


print(sin_der(X))
print(all(np.isclose(sin_der(X), np.cos(X), rtol=10**(-4)))) # On teste si les listes correspondent bien, all vérifiant si tous les éléments de la liste sont True ou non


def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
    approx = np.zeros(len(X))
    for x in range(len(X)) :
        for j in range(len(derivatives)) :
            approx[x] += (derivatives[j]/factorial(j)*X[x]**j)
    return approx

""" Test """

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)

""" Affichage """
"""
plt.plot(X, taylor1(X , sinus10), color = 'blue', label = 'taylor1')
plt.plot(X , np.sin(X) , color = 'red', label = 'sinus')
plt.title(" Comparaison des fonctions ")
plt.show()
"""

def taylor2(X, f, n):
    """
        X : le domaine
        f : la fonction (sur scalaire)
        n : le degré
    
        Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
        """
     
    approx = np.zeros_like(X)
    deriv = []
    g = f
    for i in range(1,n+1) :
        deriv.append(g(0.0))
        g =  grad(g)
    return taylor1(X,deriv)
    

# exp à l'ordre 2 : 1 + x + x^2/2
#assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)


plt.plot( X , Y , color = 'blue')
plt.plot( X, Y2, color = 'red')
plt.title( ' Comparaison')
plt.show()

# J'ai refait un test avec l'exponentielle et l'assertion sera fausse à cause de l'écart sur les bords.

# 2 ) L'approximation est acceptable sur [-4,4]
""" Tracé de cos et son appro au degré 0 """

plt.plot( X,  np.cos(X))
plt.plot(X , [1 for _ in range(len(X))])
plt.show()

# On remarque que ça ne marche pas (cohérent car 1 est de période nulle)

#3.3 

def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2 * np.sin(X) + np.cos(X/4)

Y3 = custom(X)
Y4 = taylor2(X, custom, 14)
plt.plot(X, Y3, c = 'r')
plt.plot(X,Y4,c='b')
plt.show()


