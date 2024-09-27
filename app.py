import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.impute import KNNImputer
import json
import os
import gc
import io  # Import for handling in-memory files
import base64  # Importation du module base64


# Changer la couleur de fond de la page
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe4c4;  /* Bleu ciel */
    }
    .info-box {
        background-color: #56bcd2;  /* Fond bleu */
        color: #FFFFFF;  /* Blanc clair */
        border: 2px solid #FFFFFF;  /* Bordure blanche */
        padding: 15px;
        font-size: 16px;
        border-radius: 5px; /* Coins arrondis */
        width: 100%; /* Ajuste la largeur du cadre */
        box-sizing: border-box; /* Inclut le padding et la bordure dans la largeur totale */
        margin-bottom: 20px; /* Espacement entre les sections */
    }
    .stButton>button {
        background-color: #0000FF;  /* Bleu pour les boutons */
        color: #FFFFFF;  /* Blanc clair pour le texte des boutons */
        border: 2px solid #FFFFFF;  /* Bordure blanche pour les boutons */
        border-radius: 5px; /* Coins arrondis pour les boutons */
        padding: 10px 20px;
        font-size: 16px;
    }
    @media (max-width: 768px) {
        .info-box {
            padding: 10px;
            font-size: 14px;
        }
        .stButton>button {
            padding: 8px 16px;
            font-size: 14px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Titre stylisé
st.markdown(
    """
    <div style="width: 100%; padding:10px;">
        <h1 style="color: #000000; text-align: center; margin: 0;">Applications Web pour l'Analyse de Données et le Machine Learning</h1>
    </div>
    """, unsafe_allow_html=True)

# Contrôle d'affichage pour Documentation et Contact
show_documentation = st.checkbox("Documentation", value=False)
show_contact = st.checkbox("Contact", value=False)

# Affichage de la Documentation
if show_documentation:
    st.markdown(
        """
        <div class="info-box">
            <h2>Documentation</h2>
            <p>Cette application web a été développée en 2024 par M. Mahamat Hassan Issa, étudiant en Master Big Data et Statistique à l’Université Paris-Dauphine. Elle fournit une interface interactive pour le traitement des données et la création de modèles de machine learning. Grâce à cette application, les utilisateurs peuvent charger des fichiers de données, explorer et analyser ces données, et entraîner divers modèles de machine learning pour faire des prédictions. Voici une vue d'ensemble des fonctionnalités principales :</p>
            <h4>Chargement des Fichiers</h4>
            <p>Permet le téléchargement et la lecture de différents types de fichiers (csv, txt, xlsx, xls, parquet, json).</p>
            <h4>Analyse Univariée</h4>
            <p>Les utilisateurs peuvent sélectionner une colonne pour générer des graphiques d’histogrammes ou des diagrammes en barres, selon que les données sont quantitatives ou qualitatives.</p>
            <h4>Gestion des Doublons</h4>
            <p>Affichage du nombre de doublons présents dans les données.<br>
            Option pour supprimer les doublons si souhaités.</p>
            <h4>Gestion des Données Manquantes </h4>
            <p>Options disponibles pour traiter les valeurs manquantes : suppression de lignes/colonnes, remplacement par la moyenne, la médiane, le mode, imputation par KNN ou une valeur spécifique.</p>
            <h4>Heatmap des Variables Numériques</h4>
            <p>Affiche une heatmap de la matrice de corrélation pour les variables numériques du dataframe.
            La heatmap aide à visualiser les relations entre les variables numériques.</p>
            <h4>Traitement des Dates </h4>
            <p>Détecte et convertit les colonnes de dates en formats exploitables.</p>
            <h4>Encodage des Variables Catégoriques </h4>
            <p>Permet l'encodage des variables catégoriques avec LabelEncoder et affiche le résultat.</p>
            <h4>Préparation des Données pour le Machine Learning </h4>
            <p>Permet la sélection des caractéristiques (features) et de la cible (target).
            Propose différents modèles de machine learning (régression linéaire, régression logistique, Random Forest, Decision Tree, SVM) pour l'entraînement et la prédiction.</p>
            <h4>Évaluation des Modèles </h4>
            <p>Pour les modèles de régression : affichage de MSE, RMSE, et R².
             Pour les modèles de classification : affichage de la matrice de confusion, du rapport de classification, et des courbes ROC (pour les classifications binaires et multiclasses).</p>
            <h4>Prédictions </h4>
            <p> Affichage des prédictions générées par le modèle sélectionné.<br>
            Option de téléchargement des prédictions en format CSV.</p>

           
        </div>
        """, unsafe_allow_html=True)

# Affichage du Contact
if show_contact:
    st.markdown(
        """
        <div class="info-box">
            <h2>Pour toute question ou assistance, veuillez contacter M. Mahamat Hassan Issa :</h2>
            <p>Email : mhtissaymii@yahoo.fr</p>
            <p>Téléphone : +33 745179907</p>
            <p>Visitez notre <a href="https://data-formation.net/" target="_blank" style="color: #F0F8FF; text-decoration: underline;">site web</a></p>
        </div>
        """, unsafe_allow_html=True)


st.subheader("Téléchargez un fichier pour le traitement")

file_type = st.selectbox("Choisissez le type de fichier", ["csv", "txt", "xlsx", "xls", "parquet", "json"])
uploaded_file = st.file_uploader("Choisissez un fichier", type=[file_type])

if uploaded_file:
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type == 'parquet':
        df = pd.read_parquet(uploaded_file)
    elif file_type == 'json':
        df = pd.read_json(uploaded_file, lines=True)
    elif file_type == 'txt':
        df = pd.read_csv(uploaded_file, delimiter='\t')
    elif file_type in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)

    if df is not None:
        st.write("Aperçu des données", df.head())
        st.write(f"Taille des données : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        # Analyse Univariée
        all_columns = df.columns.tolist()
        st.subheader("Analyse Univariée")
        analysis_column = st.selectbox("Sélectionner une colonne pour l'analyse univariée", all_columns)
        plot_type = st.radio("Choisissez le type de graphique", ["Histogramme pour les variables quantitatives", "Diagramme en barres pour les variables qualitatives"])

        if analysis_column:
            fig, ax = plt.subplots()

            if plot_type == "Diagramme en barres pour les variables qualitatives":
                if not pd.api.types.is_categorical_dtype(df[analysis_column]):
                    df[analysis_column] = df[analysis_column].astype('category')
                sns.countplot(x=df[analysis_column], ax=ax)
                st.pyplot(fig)

            elif plot_type == "Histogramme pour les variables quantitatives" and pd.api.types.is_numeric_dtype(df[analysis_column]):
                sns.histplot(df[analysis_column], kde=True, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Veuillez choisir un type de graphique compatible avec le type de données sélectionné.")

        # Afficher le nombre de doublons
        st.subheader("Données en Doublon")
        duplicates = df.duplicated().sum()
        st.write(f"Nombre de doublons : {duplicates}")

        if duplicates > 0:
            remove_duplicates = st.selectbox("Choisissez ce que vous voulez faire avec les doublons", ["Aucun", "Supprimer les doublons"])
            if remove_duplicates == "Supprimer les doublons":
                df = df.drop_duplicates()
                st.write("Doublons supprimés.")

          # Gestion des données manquantes par colonne
        st.subheader('Gestion des valeurs manquantes')

        missing_data_options = [
            "Aucune", 
            "Supprimer les lignes", 
            "Supprimer les colonnes", 
            "Remplacer par la moyenne", 
            "Remplacer par la médiane", 
            "Remplacer par le mode", 
            "KNN Imputer", 
            "Remplacer par une valeur spécifique"
        ]

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                st.write(f'Colonne: {col} - {df[col].isnull().sum()} valeurs manquantes')
                missing_data_option = st.selectbox(f"Choisissez une option pour la colonne {col}", missing_data_options, key=col)

                if missing_data_option == "Supprimer les lignes":
                    df = df.dropna(subset=[col])
                elif missing_data_option == "Supprimer les colonnes":
                    df = df.drop(columns=[col])
                elif missing_data_option == "Remplacer par la moyenne":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif missing_data_option == "Remplacer par la médiane":
                    df[col].fillna(df[col].median(), inplace=True)
                elif missing_data_option == "Remplacer par le mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif missing_data_option == "KNN Imputer":
                    knn_imputer = KNNImputer(n_neighbors=5)
                    df[col] = knn_imputer.fit_transform(df[[col]])
                elif missing_data_option == "Remplacer par une valeur spécifique":
                    value = st.number_input(f"Entrez la valeur de remplacement pour {col}", value=0)
                    df[col].fillna(value, inplace=True)

        st.write("Données après gestion des valeurs manquantes:")
        st.write(df.head())

        # Détection des colonnes de date potentielles
        possible_date_columns = df.select_dtypes(include=[np.datetime64, 'datetime', 'object']).columns.tolist()

        if possible_date_columns:
            date_columns = st.multiselect("Sélectionner les colonnes de date à traiter", possible_date_columns)

            if date_columns:
                for col in date_columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')  # Conversion en datetime
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_hour'] = df[col].dt.hour

                st.write(f"Les colonnes de date suivantes ont été traitées : {date_columns}")
            else:
                st.write("Aucune colonne de date sélectionnée")
        else:
            st.write("Aucune colonne de date détectée dans les données")

        # Afficher les premières lignes du dataframe pour vérifier les colonnes extraites
        st.write(df.head())

        # Affichage de la Heatmap des variables numériques
        st.subheader("Heatmap des Variables Numériques")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            correlation_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Aucune variable numérique détectée pour la Heatmap.")

        # Étape 1 : Sélectionner les Variables Catégoriques à Encoder
        st.subheader("Nombre de Modalités pour Chaque Variable Catégorique")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            num_modalities = {col: df[col].nunique() for col in categorical_cols}
            st.write(pd.DataFrame(num_modalities.items(), columns=['Variable', 'Nombre de Modalités']))
        else:
            st.info("Aucune variable catégorique détectée.")

        st.subheader("Sélectionner les Variables Catégoriques à Encoder")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            columns_to_encode = st.multiselect("Sélectionner les colonnes à encoder", ["Aucune"] + categorical_cols)

            if "Aucune" in columns_to_encode or len(columns_to_encode) == 0:
                st.info("Aucune colonne sélectionnée pour l'encodage.")
            else:
                le = LabelEncoder()
                for col in columns_to_encode:
                    df[col] = le.fit_transform(df[col].astype(str))
                st.write("Données après encodage des colonnes sélectionnées", df.head())
        else:
            st.warning("Aucune variable catégorique détectée pour l'encodage.")
            columns_to_encode = []

        # Sélectionner les Features et la Cible
        st.subheader("Sélectionner les Features et la Cible")
        all_columns = df.columns.tolist()

        target = st.selectbox("Sélectionner la variable cible", all_columns)
        features = st.multiselect("Sélectionner les variables explicatives", [col for col in all_columns if col != target])

        if target in features:
           st.warning(f"Vous avez sélectionné la variable cible ({target}) comme feature. Veuillez la retirer de la liste des variables explicatives.")
           features.remove(target)

        st.subheader("Choisissez le type de Modèle")
        model_type = st.selectbox("Type de Modèle", ["Régression", "Classification Binaire", "Classification Multiclasses"])

        # Ajouter la sélection du modèle en fonction du type de modèle choisi
        if model_type == "Régression":
           selected_model = st.selectbox("Choisissez un modèle de régression", 
                                  ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "SVR"])
        elif model_type == "Classification Binaire":
            selected_model = st.selectbox("Choisissez un modèle de classification binaire", 
                                  ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "SVC"])
        elif model_type == "Classification Multiclasses":
           selected_model = st.selectbox("Choisissez un modèle de classification multiclasses", 
                                  ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "SVC"])

        # Ajouter un slider pour choisir la taille de l'ensemble de test
        test_size = st.slider("Taille de l'ensemble de test (%)", 10, 50, 20) / 100

        # Sélection du nombre de plis pour la validation croisée
        cv_folds = st.slider("Nombre de plis (folds) pour la validation croisée", 2, 10, 5)

        # Bouton pour exécuter le modèle
        if st.button("Entraîner le modèle"):
           if target and features:
              st.write(f"Variable cible : {target}")
              st.write(f"Variables explicatives : {features}")

              X = df[features]
              y = df[target]

              # Split data avec la taille de test choisie par l'utilisateur
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

              # Sélectionner le modèle basé sur le choix de l'utilisateur
              if model_type == "Régression":
                 if selected_model == "Linear Regression":
                    model = LinearRegression()
                 elif selected_model == "Random Forest Regressor":
                    model = RandomForestRegressor()
                 elif selected_model == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                 elif selected_model == "SVR":
                  model = SVR()

              elif model_type == "Classification Binaire":
                 if selected_model == "Logistic Regression":
                   model = LogisticRegression()
                 elif selected_model == "Random Forest Classifier":
                  model = RandomForestClassifier()
                 elif selected_model == "Decision Tree Classifier":
                  model = DecisionTreeClassifier()
                 elif selected_model == "SVC":
                   model = SVC(probability=True)

              elif model_type == "Classification Multiclasses":
                  if selected_model == "Logistic Regression":
                     model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
                  elif selected_model == "Random Forest Classifier":
                     model = RandomForestClassifier()
                  elif selected_model == "Decision Tree Classifier":
                     model = DecisionTreeClassifier() 
                  elif selected_model == "SVC":
                     model = SVC(probability=True)

               # Validation croisée avec le nombre de plis choisi par l'utilisateur
              kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
              cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

              st.write("Scores de validation croisée : ", cv_scores)
              st.write("Score moyen de validation croisée : ", np.mean(cv_scores))

              # Entraîner le modèle
              model.fit(X_train, y_train)
            
              # Prédictions
              y_pred = model.predict(X_test)
             
              # Créer un DataFrame pour les prédictions
              predictions_df = pd.DataFrame({
                   'Index': X_test.index,
                    'Valeurs Réelles': y_test,
                    'Valeurs Prédites': y_pred
              })

              if model_type == "Régression":
                # Afficher les métriques de régression
                 st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
                 st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")  # Ajout du RMSE
                 st.write(f"R2 Score: {r2_score(y_test, y_pred)}")
                # Graphique de régression
                 st.subheader("Graphique de régression : Valeurs prédites vs Valeurs réelles")
                 fig, ax = plt.subplots()
                 sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                 ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                 ax.set_xlabel('Valeurs réelles')
                 ax.set_ylabel('Valeurs prédites')
                 ax.set_title('Valeurs réelles vs. Valeurs prédites')
                 st.pyplot(fig)
              else:
                  # Afficher les métriques de classification
                  st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
                  st.write("Confusion Matrix:")

                  # Matrice de confusion améliorée
                  cm = confusion_matrix(y_test, y_pred)
                  fig, ax = plt.subplots(figsize=(8, 6))
                  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                  ax.set_xlabel('Valeurs prédites')
                  ax.set_ylabel('Valeurs réelles')
                  ax.set_title('Matrice de Confusion')
                  st.pyplot(fig)

                  # Afficher le classification report bien formaté
                  st.write("Classification Report:")
                  report = classification_report(y_test, y_pred, output_dict=True)
                  report_df = pd.DataFrame(report).transpose()
                  st.dataframe(report_df.style.format("{:.2f}"))

                  # Courbe ROC pour la classification binaire
                  if model_type == "Classification Binaire":
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)

                  # Courbe ROC pour la classification multiclasses
                  elif model_type == "Classification Multiclasses":
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for i in range(len(np.unique(y_test))):
                        y_test_bin = np.where(y_test == i, 1, 0)
                        y_prob = model.predict_proba(X_test)[:, i]
                        fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

                    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (Multiclass)')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                  predictions_df = pd.DataFrame({'Réel': y_test, 'Prédit': y_pred})

                  # Afficher les prédictions
                  st.write("Prédictions", predictions_df)

                  # Préparer le CSV pour le téléchargement
                  buffer = io.StringIO()
                  predictions_df.to_csv(buffer, index=False)
                   # Bouton de téléchargement stylisé
                  st.markdown(f"""
                    <a href="data:file/csv;base64,{base64.b64encode(buffer.getvalue().encode()).decode()}" 
                       download="predictions.csv" class="custom-button">
                       Télécharger les prédictions en CSV
                    </a>
                  """, unsafe_allow_html=True)
    

gc.collect()
