import streamlit as st

# Main page content

st.markdown(
        """
        <div style="text-align: center; font-size: 1.5rem; color: gray;">
            Jedha Evaluation Tyres
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    "<div style='text-align: center; font-size: 1.1rem; color: gray;'>A Convolutional Neural Network Project</div>",
    unsafe_allow_html=True
)

st.write("")

col1, col2, col3 = st.columns([1, 2, 1])  # col2 est au centre, plus large

with col2:
    st.image("logo/Logo_JET.png", use_container_width=True)

st.write("")

st.markdown("""
            L'objectif de ce projet est de déployer un modèle de *deep learning* permettant de contrôler la qualité des pneus à partir d'une image importée.
            La classification est la la suivante:

            🚨🛞Contrôle de la qualité du pneu par image.
            * Bon état (apte à rouler) ✅
            * Mauvaise état (pas apte à rouler/à changer) ❌

            Les informations sur les données utilisées se trouvent dans la page suivante, `Dataset`.
            Les différentes informations sur le modèle de baseline, le plus simple et correct pour des premiers résultat,
            ainsi que sur le meilleur modèle obtenu pour effectuer les prédictions se trouvent sur la page `Model`.
            La page `Predictions` vous permet de charger une photo de pneu et d'obtenir une prédiction sur sa qualité, avec le taux de confiance du maodèle.

            Enfin, pour un usage future à plus grande échelle, on pourrait imaginer la mise en place de capteurs industrialisés, 
            qui se placeraient en face de chaque roue et à chaque démarage du véhicule, le capteur prendrait une photo du pneu. 
            Puis avec le `JET model` implémenter à l"intérieur, il calculerait l'état du pneu et pourrait renvoyer l'information
            au conducteur sur le tableau de bord, indiquant si tout va bien ✅ où s'il y a un danger ❌ et un changement à faire.

            """)
# st.markdown("Accueil")
# st.sidebar.markdown("Accueil")

