import streamlit as st
import autograd.numpy as np
from autograd import grad

st.title("Approximation de Taylor")
st.subheader('Nous allons approximer des fonctions avec leur développement taylorien')
st.latex(r''' P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k''')
#valeur = st.slider("Choisissez une valeur", 0, 10)
#st.write("Vous avez choisi :", valeur)

# Le script est exécuté de nombreuses fois, ce qui peut entraîner un lag de l'ordinateur


