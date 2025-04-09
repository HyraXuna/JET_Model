# import streamlit as st
# import streamlit.components.v1 as components

# # bootstrap 4 collapse example
# components.html(
#     """
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
#     <div id="accordion">
#       <div class="card">
#         <div class="card-header" id="headingOne">
#           <h5 class="mb-0">
#             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
#             Tyre Quality Classification - Digital images of defective and good condition tyres 
#             </button>
#           </h5>
#         </div>
#         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
#           <div class="card-body">
#             <ul>
#                 <li>https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data</li>
#                 <li>By Chirag CHAUHAN - mis à jour il y a 2 ans</li>
#                 <li>Description : L'ensemble de données contient 1854 images numériques de pneus, classées en deux catégories : défectueux et en bon état. Chaque image est au format numérique et représente un pneu. Les images sont étiquetées selon leur état, c'est-à-dire si le pneu est défectueux ou en bon état.</b>

# Cet ensemble de données peut être utilisé pour diverses applications d'apprentissage automatique et de vision par ordinateur, telles que la classification d'images et la détection d'objets. Les chercheurs et les praticiens des secteurs des transports, de l'automobile et du contrôle qualité peuvent l'utiliser pour entraîner et tester leurs modèles afin d'identifier l'état des pneus à partir d'images numériques. Cet ensemble de données constitue une ressource précieuse pour développer et évaluer les performances d'algorithmes de détection automatique des pneus défectueux.

# Cet ensemble de données pourrait également contribuer à améliorer le processus de contrôle qualité de l'industrie du pneumatique et à réduire les risques d'accidents dus à des pneus défectueux. Sa disponibilité faciliterait le développement de systèmes d'inspection plus précis et plus efficaces pour la production de pneumatiques.
#             </ul>
#             <div style="margin-top: 10px;">
#                 <span style="display: inline-block; background-color: #e0f3ff; color: #007bff; padding: 5px 10px; margin: 4px; border-radius: 20px;">
#                     Image
#                 </span>
#                 <span style="display: inline-block; background-color: #fff3cd; color: #856404; padding: 5px 10px; margin: 4px; border-radius: 20px;">
#                     Computer Vision
#                 </span>
#                 <span style="display: inline-block; background-color: #e2f7e1; color: #2e7d32; padding: 5px 10px; margin: 4px; border-radius: 20px;">
#                     Deep Learning
#                 </span>
#                 <span style="display: inline-block; background-color: #fce4ec; color: #ad1457; padding: 5px 10px; margin: 4px; border-radius: 20px;">
#                     Image Classification
#                 </span>
#                 <span style="display: inline-block; background-color: #d1ecf1; color: #0c5460; padding: 5px 10px; margin: 4px; border-radius: 20px;">
#                     Binary Classification
#                 </span>
#             </div>
#           </div>
#         </div>
#       </div>
#       <div class="card">
#         <div class="card-header" id="headingTwo">
#           <h5 class="mb-0">
#             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
#             Collapsible Group Item #2
#             </button>
#           </h5>
#         </div>
#         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
#           <div class="card-body">
#             Collapsible Group Item #2 content
#           </div>
#         </div>
#       </div>
#     </div>
#     """,
#     height=600,
# )

import streamlit as st
import streamlit.components.v1 as components

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" crossorigin="anonymous"></script>

    <div id="accordion">
      <div class="card">
        <div class="card-header" id="headingOne">
          <h5 class="mb-0">
            <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
              Tyre Quality Classification – Digital images of defective and good condition tyres
            </button>
          </h5>
        </div>

        <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
          <div class="card-body">
            <ul>
              <li><a href="https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data" target="_blank">
              https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data</a></li>
              <li>By Chirag CHAUHAN – mis à jour il y a 2 ans</li>
              <li>
                <strong>Description :</strong>
                <p>L'ensemble de données contient 1854 images numériques de pneus, classées en deux catégories : défectueux et en bon état. Chaque image est au format numérique et représente un pneu. Les images sont étiquetées selon leur état.</p>

                <p>🧠 Cet ensemble de données peut être utilisé pour diverses applications d'apprentissage automatique et de vision par ordinateur, telles que la classification d'images et la détection d'objets. Les chercheurs et praticiens dans les domaines du transport, de l'automobile et du contrôle qualité peuvent l’utiliser pour entraîner et tester des modèles.</p>

                <p>🔍 Il constitue une ressource précieuse pour développer et évaluer des algorithmes de détection automatique des pneus défectueux.</p>

                <p>✅ Cela pourrait améliorer le contrôle qualité dans l’industrie du pneumatique et réduire les risques d’accidents dus à des pneus défectueux.</p>
              </li>
            </ul>

            <div style="margin-top: 10px;">
                <span style="display: inline-block; background-color: #e0f3ff; color: #007bff; padding: 5px 10px; margin: 4px; border-radius: 20px;">
                    Image
                </span>
                <span style="display: inline-block; background-color: #fff3cd; color: #856404; padding: 5px 10px; margin: 4px; border-radius: 20px;">
                    Computer Vision
                </span>
                <span style="display: inline-block; background-color: #e2f7e1; color: #2e7d32; padding: 5px 10px; margin: 4px; border-radius: 20px;">
                    Deep Learning
                </span>
                <span style="display: inline-block; background-color: #fce4ec; color: #ad1457; padding: 5px 10px; margin: 4px; border-radius: 20px;">
                    Image Classification
                </span>
                <span style="display: inline-block; background-color: #d1ecf1; color: #0c5460; padding: 5px 10px; margin: 4px; border-radius: 20px;">
                    Binary Classification
                </span>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
              Collapsible Group Item #2
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #2 content
          </div>
        </div>
      </div>
    </div>
    """,
    height=700,
)
