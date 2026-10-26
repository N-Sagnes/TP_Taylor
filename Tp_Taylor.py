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
## 3.3
def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2 * np.sin(X) + np.cos(X/4)

## 3.2 
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


### 4
## 4.2

ln = lambda x : np.log(1+x)
FONCTIONS = { 'sinus' : np.sin , 'cosinus' : np.cos , 'exp' : np.exp , 'ln' : ln , 'custom' : custom}
## 4.3
with st.sidebar :
    st.selectbox( 'Sélection de la fonction', FONCTIONS , key="choix") # On attribue une clé pour pouvoir appeler la valeur dans les tracés
    st.slider('Choix de l_ordre ', 0,10, value = 3 , key = "ordre")
    st.slider('Largeur de l_intervalle étudié' , 2.0 , 20.0 ,value = 12.6 , step = 0.1, key = "intervalle")


## 4.4
f = st.session_state.choix
X = np.linspace(-st.session_state.intervalle/2,st.session_state.intervalle/2,400)
Y = FONCTIONS[f](X)
Y_T = taylor2(X,FONCTIONS[f],st.session_state.ordre)
fig,ax = plt.subplots()
ax.plot(X, Y, color ='r', label = "Fonction choisie")
ax.plot(X, Y_T, color = 'b', label = "Approximation de Taylor")
st.pyplot(fig)

# A n fixé, la qualité de notre approximation diminue lorsque l'on s'éloigne de 0.
# Pour ln(1+X), au delà de x=1, si on augmente n, on observe que notre approx n'est en fait qu'un plateau.

### 5
## 5.1
with st.sidebar :
    st.select_slider('Taux de coïncidence', 
                     options = [10**(-2),10**(-3),10**(-4),10**(-5),10**(-6)],
                       value = 10**(-3),
                       key = "seuil")

def taux_coïncidence(Y,Y_T,seuil) :
    return np.mean(np.abs(Y-Y_T)<seuil)
taux = taux_coïncidence(Y,Y_T , st.session_state.seuil)

st.write("Taux de coïncidence", )

##5.2

st.checkbox('Afficher l_erreur', key='err') #checkbox permet de cocher où non la case d'affichage 

if st.session_state.err: # boucle if pour agir si la case est cochée
    erreur = np.abs(Y - Y_T) #on calcule l'erreur
    fig_erreur, ax_erreur = plt.subplots() 
    ax_erreur.plot(X, erreur, label="Erreur absolue", color="green") # On plot l'erreur entre la courbe et l'approximation
    ax_erreur.set_yscale("log") # on transforme l'échelle des ordonnées en échelle logarithmique
    ax_erreur.legend() #on affiche la légende
    st.pyplot(fig_erreur)


### 6.

if "ordres" not in st.session_state: # session_state permet de se souvenir de ce qui s'est passé dans l'éxécution précédente
    st.session_state.ordres = []

with st.sidebar:
    if st.button("Ajouter cet ordre à la comparaison"): # on crée le bouton dans la barre latérale
        n = st.session_state.ordre # on donne a la valeur n l'ordre choisis
        if n not in st.session_state.ordres: # on évite les doublons
            st.session_state.ordres.append(n)

with st.sidebar:
    if st.button("Réinitialiser la comparaison"): # on crée la manière de recommencer la liste
        st.session_state.ordres = []

Y = FONCTIONS[f](X) # Calcul de la fonction

fig, ax = plt.subplots() # on trace la fonctions et les approximations
ax.plot(X, Y, label="Fonction f", color="black", linewidth=2)

for x in st.session_state.ordres: 
    Y_taylor = taylor2(X, FONCTIONS[f], x) # on calcul les approximation pour tout les ordres qui ont été stockés dans la liste
    ax.plot(X, Y_taylor, label=f"Taylor ordre {x}") #on affiche sur le même graphique chaque courbe pour chaque ordre


ax.set_title("Comparaison des approximations de Taylor")
ax.legend() # afiiche la légende
st.pyplot(fig)

st.write("Ordres sélectionnés :", st.session_state.ordres) # Affichage des ordres que l'on a choisi.





    