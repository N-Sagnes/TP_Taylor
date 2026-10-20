import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

st.title("Approximation de Taylor")
st.subheader('Nous allons approximer des fonctions avec leur développement taylorien issu des séries entières')
st.latex(r''' P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k''')
#valeur = st.slider("Choisissez une valeur", 0, 10)
#st.write("Vous avez choisi :", valeur)

# Le script est exécuté de nombreuses fois, ce qui peut entraîner un lag de l'ordinateur
def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2 * np.sin(X) + np.cos(X/4)

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

ln = lambda x : np.log(1+x)

FONCTIONS = { 'sinus' : np.sin , 'cosinus' : np.cos , 'exp' : np.exp , 'ln' : ln , 'custom' : custom}
with st.sidebar :
    st.selectbox( 'Sélection de la fonction', FONCTIONS )
    st.slider('Choix de l_ordre ', 0,10, value = 3 )
    st.slider('Largeur de l_intervalle étudié' , 2.0 , 20.0 ,value = 12.6 , step = 0.1)


