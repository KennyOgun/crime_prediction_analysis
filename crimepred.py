import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc, confusion_matrix, roc_auc_score
import pickle
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
import lime.lime_tabular
import shap
import dill as pickle
import altair as alt
import xgboost as xgb



########################  WELCOME PAGE   ##############################
# Aesthetic settings
st.set_page_config(page_title="Crime Prediction and Classification Tool", layout="wide")

# Welcome Page
st.title("Crime Prediction and Classification Tool")
st.subheader("Developed by Kehinde Ogundana")
st.write("""
This ML project visualizes pre-trained models - KNN, Random Forest, XGBoost for crime classification. 
The goal is to assist law enforcement in reducing crimes, optimizing resource allocation, 
and minimizing citizens' exposure to crimes.
""")

st.markdown("Click **Data Exploration and Visualization** to get started.")

############################ UPLOAD DATA SECTION  ##########################################

# Sidebar for user interaction
st.sidebar.title("Crime Data Analysis")
uploaded_file = st.sidebar.file_uploader("Upload your data here (CSV)", type="csv")

# Initialize df as None
df = None

# Check if a file has been uploaded
if uploaded_file is not None:
    # If the file is uploaded, read it into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")
    # Display first 5 rows of the uploaded data
    st.write("First 5 rows of the uploaded data:")
    st.write(df.head(5))
else:
    st.sidebar.info("No dataset loaded. Please upload a dataset to proceed.")



######################### DATA EXPLORATION ##########################################

# Sidebar for data exploration
st.sidebar.subheader("Data Exploration and Visualization")
 # Ensure file is uploaded before proceeding
if df is not None:
    exploration_option = st.sidebar.selectbox(
        "Choose a visualization:",
        [
            "Top 15 'Primary Crime' Types",
            "Crime Distribution in Top 5 Crime-Prone Community Areas",
            "Arrests vs Non-Arrests",
            "Heatmap for 'THEFT' and 'ASSAULT'",
        ]
)


    st.write("### Data Exploration and Visualization Section")
# Top 15 'Primary Type' Crimes Visualization
    if exploration_option == "Top 15 'Primary Crime' Types":
        st.write("### Top 15 Primary Crime Type in Chicago")
        top_15_primary_types = df['primary_type'].value_counts().nlargest(15)

       
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_15_primary_types.values, y=top_15_primary_types.index, palette='viridis', ax=ax)
        ax.set_title("Top 15 Primary Crime Type in Chicago")
        st.pyplot(fig)
        st.write("**The bar chart shows that theft is the most prevalent crime," 
            "followed by battery, indicating a significant skew towards non-violent offenses.**")


# Crime Distribution in Top 5 Crime-Prone Community Areas
    elif exploration_option == "Crime Distribution in Top 5 Crime-Prone Community Areas":

        community_area_mapping = {8: 'Near North Side', 25: 'Austin', 28: 'Near West Side', 32: 'Loop', 43: 'South Shore'}
        # Get the top 5 community areas based on crime count
        top5_community_areas_index = df['community_area'].value_counts().head(5).index
        top5_community_areas = df[df['community_area'].isin(top5_community_areas_index)]

        # Get the top 5 primary types of crimes
        top5_primary_types = df['primary_type'].value_counts().nlargest(5).index
        top5_and_top5_crime_df = top5_community_areas[top5_community_areas['primary_type'].isin(top5_primary_types)]

        # Pivot table for the top 5 community areas and top 5 crime types
        Top5_pivot = top5_and_top5_crime_df.pivot_table(index='community_area', columns='primary_type', aggfunc='size', fill_value=0)

        # Normalize to get percentages
        Top5_pivot = Top5_pivot.loc[Top5_pivot.sum(axis=1).sort_values(ascending=False).index]
        Top5_pivot = Top5_pivot.pipe(lambda df: (df.T / df.sum(axis=1)).T * 100)

        # Replace community area numbers with community area names
        Top5_pivot.index = Top5_pivot.index.map(community_area_mapping)



        # Reset index for better handling with Altair
        Top5_pivot = Top5_pivot.reset_index().melt(id_vars='community_area', var_name='primary_type', value_name='Percentage')

        # Altair chart
        chart = alt.Chart(Top5_pivot).mark_bar().encode(
            y=alt.Y('community_area:N', sort='-x', title='Community Area'),
            x=alt.X('Percentage:Q', stack='normalize', title='Crime Proportion (%)'),
            color='primary_type:N',
            tooltip=['community_area', 'primary_type', 'Percentage']
        ).properties(
            width=800,
            height=400,
            title='Crime Distribution by Primary Type in Top 5 Community Areas'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_legend(
            titleFontSize=14,
            labelFontSize=12
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

        
        st.write("**The bar chart highlights the community areas with the highest crime rates."
        " For non-violent crimes, the Loop and Near North Side recorded the highest levels of theft," 
        "at 61% and 57% respectively. In contrast, Austin leads in violent crimes, "
        "with 34% of incidents classified as violent (Battery), followed closely by South Shore at 33%.**")
             
                              
# Crime Count (Arrest & Domestic)
    elif exploration_option == "Arrests vs Non-Arrests":
        top_15_primary_types = df['primary_type'].value_counts().nlargest(15)
        top_15_crime_df = df[df['primary_type'].isin(top_15_primary_types.index)]
        arrest_data = top_15_crime_df.groupby(['primary_type', 'arrest']).size().unstack(fill_value=0)

        st.write("### Arrests vs Non-Arrests for Top 15 Primary Crime Type")
        fig, ax = plt.subplots(figsize=(12, 8))
        arrest_data.plot(kind='barh', stacked=True, color=['salmon', 'lightblue'], ax=ax)
        ax.set_title("Arrests vs Non-Arrests for Top 15 Primary Crime Type in Chicago")
        st.pyplot(fig)

        st.write("**The bar chart highlights a notable disparity between arrest and non-arrest outcomes."
         "Less than 13% of crimes resulted in arrests, pointing to a considerable gap in law enforcement effectiveness.**")
            

# Heatmap for 'THEFT' and 'ASSAULT'
    elif exploration_option == "Heatmap for 'THEFT' and 'ASSAULT'":
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Heatmap for THEFT")
            theft_df = df[df['primary_type'] == 'THEFT'].dropna(subset=['latitude', 'longitude'])
            theft_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
            heat_data = [[row['latitude'], row['longitude']] for index, row in theft_df.iterrows()]
            HeatMap(heat_data).add_to(theft_map)
            st.components.v1.html(theft_map._repr_html_(), height=500)

        with col2:
            st.write("### Heatmap for ASSAULT")
            assault_df = df[df['primary_type'] == 'ASSAULT'].dropna(subset=['latitude', 'longitude'])
            assault_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
            heat_data = [[row['latitude'], row['longitude']] for index, row in assault_df.iterrows()]
            HeatMap(heat_data).add_to(assault_map)
            st.components.v1.html(assault_map._repr_html_(), height=500)
        st.write("**Heatmap: Theft and Assault**")
                  


# ############################### DATA PREPROCESSING SECTION ###################################################
 # Ensure file is uploaded before proceeding
st.sidebar.subheader("Data Preprocessing")
if df is not None:
    preprocess_option = st.sidebar.selectbox("Select preprocessing step:", ["Label Encoding", "Feature Selection, Data Splitting and Scaling"])

    if preprocess_option == "Label Encoding":
        st.write("# Label Encoding Section")
        # Convert boolean columns to 0 and 1
        bool_columns = df.select_dtypes(include='bool').columns
        df[bool_columns] = df[bool_columns].astype(int)

        # Apply Label Encoding to categorical columns
        le = LabelEncoder()
        categorical_columns = ["description", "crime_category", "loc_desc_cat", "primary_type", "time_of_day"]

        # Create a copy of the dataframe for encoded data
        df_enc = df.copy()

        for col in categorical_columns:
            if col in df.columns:
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))

        st.write("### Categorical features encoded successfully!")

        # Save df_enc to session state to persist it
        st.session_state['df_enc'] = df_enc

        # Define Numerical and Categorical Columns
        numeric_features = [feature for feature in df_enc.columns if df_enc[feature].dtype != 'O']
        categorical_features = [feature for feature in df_enc.columns if df_enc[feature].dtype == 'O']

        st.write(f"There are {len(numeric_features)} numeric features: {numeric_features}")
        st.write(f"There are {len(categorical_features)} categorical features: {categorical_features}")

    elif preprocess_option == "Feature Selection, Data Splitting and Scaling":
        st.write("# Feature Selection, Data Splitting and Scaling Section")
        # Check if df_enc exists in session state
        if 'df_enc' in st.session_state:
            df_enc = st.session_state['df_enc']

            # Select features from the encoded dataframe
            selected_features = ['description', 'domestic', 'arrest', 'is_weekend', 'hour', 'year', 'longitude', 'latitude',
                                'distance_to_vacant_building', 'distance_to_police_station', 'distance_to_school', 'crime_category']

            df_selected = df_enc[selected_features]

            st.write(df_selected.head(5))

            # Split the data
            X = df_selected.drop(columns=['crime_category'])  # Features (independent variables)
            y = df_selected['crime_category']  # Target (dependent variable)

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Scale the numeric features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.write("Feature Selection ,Data split and scaled successfully!")
            st.write(f"X_train shape: {X_train_scaled.shape}")
            st.write(f"X_test shape: {X_test_scaled.shape}")
            st.write(f"y_train shape: {y_train.shape}")
            st.write(f"y_test shape: {y_test.shape}")



            # Store processed data in session state for later use
            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.success("# Data preprocessed and ready for training.")

          #  st.write("### Data Preprocessing Done Successfully!")
        else:
            st.info("Please apply Label Encoding first before proceeding to Data Splitting and Scaling.")

# ############################### MODEL SELECTION AND TRAINING SECTION #########################################

# Model Selection and Training Section
st.sidebar.subheader("Model Selection and Training")
model_option = st.sidebar.selectbox("Choose a model to train:", ["KNN", "Random Forest", "XGBoost"])
st.write("### Model Selection and Training Section")

# Ensure X_train_scaled and y_train are loaded properly
if 'X_train_scaled' in st.session_state and 'y_train' in st.session_state:
    X_train_scaled = st.session_state['X_train_scaled']
    y_train = st.session_state['y_train']

    if model_option == "KNN":
        st.subheader("KNN Model Training")
        n_neighbors = st.sidebar.selectbox("Select n_neighbors for KNN", [1, 3, 5, 7, 9])
        weights = st.sidebar.selectbox("Select weights for KNN", ["distance"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train_scaled, y_train)
        st.write(f"KNN Model trained successfully with n_neighbors={n_neighbors} and weights = {weights}")

    elif model_option == "Random Forest":
        st.subheader("Random Forest Model Training")
        n_estimators = st.sidebar.selectbox("Select n_estimators", [100, 200, 300])
        max_depth = st.sidebar.selectbox("Select max_depth", [10, 20, 30])
        min_samples_split = st.sidebar.selectbox("Select min_samples_split", [2, 5])
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train_scaled, y_train)
        st.write(f"Random Forest Model trained successfully with n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")

    elif model_option == "XGBoost":
        st.subheader("XGBoost Model Training")
        n_estimators = st.sidebar.selectbox("Select n_estimators", [100, 200, 300])
        learning_rate = st.sidebar.selectbox("Select learning_rate", [0.01, 0.1, 0.2])
        max_depth = st.sidebar.selectbox("Select max_depth", [3, 6])
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        model.fit(X_train_scaled, y_train)
        st.write(f"XGBoost Model trained successfully with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")



else:
    st.info("Data has not been preprocessed yet. Please complete the data preprocessing steps first.")



    
 # ------------------------ MODEL EVALUATION ------------------------

 ############# TEST ACCURACY, FI, RECALL, PRECISION SECTION #####

# Check if a model is trained or loaded
if 'model' in locals() or 'model' in globals():
    if model is not None:
        # Ensure test data is available in session state
        if 'X_test_scaled' in st.session_state and 'y_test' in st.session_state:
            X_test_scaled = st.session_state['X_test_scaled']
            y_test = st.session_state['y_test']

            # Model Prediction
            y_pred = model.predict(X_test_scaled)

            ########## EVALUATION METRICS ###################
            test_accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            st.subheader("Model Evaluation Metrics")
            st.write(f"**Test Accuracy:** {test_accuracy:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**Precision:** {precision:.4f}")
            st.write(f"**Recall:** {recall:.4f}")
            st.write(pd.Series(y_test).value_counts())

            ################## ROC AND AUC SCORE ####################

            # Debug: Check unique values in y_test
            st.write(f"Unique values in y_test: {y_test.unique()}")

            # ROC and AUC for Binary Classification
            if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
                y_scores = model.predict_proba(X_test_scaled)[:, 1]

                # Check for valid predicted probabilities
                if np.all(y_scores == y_scores[0]):
                    st.error("Model is predicting the same probability for all instances. AUC cannot be computed.")
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_scores)
                    roc_auc = auc(fpr, tpr)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                    st.write(f"**ROC-AUC Score:** {roc_auc:.4f}")
            else:
                st.warning("ROC-AUC is not applicable")

            ################## CONFUSION MATRIX SECTION ############################

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            cm_df = pd.DataFrame(cm, index=['Non-violent', 'Violent'], columns=['Non-violent', 'Violent'])

            # Reshape for Altair plot
            cm_df_reset = cm_df.reset_index().melt(id_vars='index')
            cm_df_reset.columns = ['Actual', 'Predicted', 'Count']

            confusion_chart = alt.Chart(cm_df_reset).mark_rect().encode(
                x='Predicted:O',
                y='Actual:O',
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['Actual', 'Predicted', 'Count']
            ).properties(
                title='Confusion Matrix',
                width=400,
                height=300
            )

            text = alt.Chart(cm_df_reset).mark_text(
                align='center',
                baseline='middle',
                fontSize=16,
                fontWeight='bold'
            ).encode(
                x='Predicted:O',
                y='Actual:O',
                text='Count:Q',
                color=alt.condition(
                    alt.datum.Count > cm.max() / 2,
                    alt.value('white'),
                    alt.value('black')
                )
            )

            final_chart = confusion_chart + text
            st.altair_chart(final_chart, use_container_width=True)

            ################# ACTUAL VS PREDICTED #####################

        st.subheader("Actual vs Predicted (Sample of 10)")

        sample_size = 10
        if len(y_pred) >= sample_size:
            sample_indices = np.random.choice(np.arange(len(y_pred)), sample_size, replace=False)
            sample_actual = y_test.iloc[sample_indices]
            sample_predictions = y_pred[sample_indices]

            plt.figure(figsize=(10, 6))
            plt.scatter(sample_indices, sample_actual, color='blue', label='Actual (0: Violent, 1: Non-Violent)', alpha=0.6)
            plt.scatter(sample_indices, sample_predictions, color='orange', label='Predicted (0: Violent, 1: Non-Violent)', alpha=0.6)
            plt.title('Actual vs Predicted (Sample of 10) - Crime Classification')
            plt.xlabel('Sample Index')
            plt.ylabel('Crime Class (0: Violent, 1: Non-Violent)')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            # Create a DataFrame with 'instance' column before 'Sample Index'
            comparison_df = pd.DataFrame({
                'Instance': np.arange(1, sample_size + 1),
                'Sample Index': sample_indices,
                'Actual': sample_actual,
                'Predicted': sample_predictions
            })

            st.subheader("Actual vs Predicted - Tabular Format")
            st.dataframe(comparison_df)
        else:
            st.warning("Not enough samples to display Actual vs Predicted.")

            ######################## PERMUTATION IMPORTANCE ######################################

################### Permutation Importance ###################



            st.subheader("Permutation Importance (Top 5 Features)")

            result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
            sorted_idx = result.importances_mean.argsort()

            if 'feature_names' in st.session_state:
                feature_names = st.session_state['feature_names']
            else:
                feature_names = [f"Feature {i}" for i in range(X_test_scaled.shape[1])]

            top_n = 5
            plt.figure(figsize=(8, 6))
            plt.barh(range(top_n), result.importances_mean[sorted_idx[-top_n:]], align='center')
            plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[-top_n:]])
            plt.title('Permutation Importance (Top 5 Features)')
            plt.xlabel('Mean Importance')
            st.pyplot(plt)



            ###################
            # Permutation Importance (Handling NumPy arrays)
        st.subheader("Permutation Importance (Top 5 Features)")

        # Assuming model, X_test_scaled, and y_test are defined and valid.
        result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)

        # Calculate sorted indices based on importances
        sorted_idx = result.importances_mean.argsort()

        # Use feature names if available
        if 'feature_names' in st.session_state:
            feature_names = st.session_state['feature_names']
        else:
            feature_names = [f"Feature {i}" for i in range(X_test_scaled.shape[1])]

        # Plotting only the top 5 features
        top_n = 5
        plt.figure(figsize=(8, 6))
        plt.barh(range(top_n), result.importances_mean[sorted_idx[-top_n:]], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[-top_n:]])
        plt.title('Permutation Importance (Top 5 Features)')
        plt.xlabel('Mean Importance')

        # Display the plot in Streamlit
        st.pyplot(plt)

         # Assuming your original data (before scaling) is a DataFrame named X_train
        if 'feature_names' not in st.session_state:
            st.session_state['feature_names'] = X_train.columns.tolist()  # Initialize feature names in session_state



            ################# LIME EXPLANATION ##############################

                  # LIME Explanation for a Random Instance
        st.subheader("LIME Explanation for a Random Instance")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            st.session_state['X_train_scaled'],  # No need for .values
            feature_names=st.session_state['feature_names'],
            class_names=['Non-violent Crime', 'Violent Crime'],
            mode='classification'
            )

        i = np.random.randint(0, st.session_state['X_test_scaled'].shape[0])
        instance_to_explain = st.session_state['X_test_scaled'][i]  # Direct access to NumPy array without .values
        explanation = explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=10)
        st.write(f"**Explanation for Instance {i}:**")
        for feature, importance in explanation.as_list():
            st.write(f"{feature}: {importance}")
        st.pyplot(explanation.as_pyplot_figure())


        # Initialize feature names from the original DataFrame before scaling
        if 'feature_names' not in st.session_state:
            st.session_state['feature_names'] = X_train.columns.tolist()  # Store feature names from DataFrame




                    ############################################### LIME Explanation for a Selected Instance
        # st.subheader("LIME Explanation for a Selected Instance")

        # # Initialize feature names from the original DataFrame before scaling
        # if 'feature_names' not in st.session_state:
        #     st.session_state['feature_names'] = X_train.columns.tolist()  # Store feature names from DataFrame

        # # Create LIME explainer
        # explainer = lime.lime_tabular.LimeTabularExplainer(
        #     st.session_state['X_train_scaled'],  # No need for .values
        #     feature_names=st.session_state['feature_names'],
        #     class_names=['Non-violent Crime', 'Violent Crime'],
        #     mode='classification'
        # )

        # # Allow the user to select an instance index
        # selected_index = st.number_input(
        #     "Select row index to explain:",
        #     min_value=0,
        #     max_value=st.session_state['X_test_scaled'].shape[0] - 1,
        #     value=0,  # Default value
        #     step=1
        # )

        # # Get the selected instance to explain
        # instance_to_explain = st.session_state['X_test_scaled'][selected_index]  # Direct access to NumPy array without .values
        # explanation = explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=10)

        # # Display explanation results
        # st.write(f"**Explanation for Instance {selected_index}:**")
        # for feature, importance in explanation.as_list():
        #     st.write(f"{feature}: {importance}")

        # # Plot the LIME explanation
        # st.pyplot(explanation.as_pyplot_figure())


            

     

            ######################## SHAP EXPLANATION ##########################

#                       # SHAP Analysis for XGBoost
#             if model_option == "XGBoost":
#                 st.subheader("SHAP Analysis")
#                 explainer_shap = shap.Explainer(model)
#                 shap_values = explainer_shap(st.session_state['X_test_scaled'])  # Get SHAP values

#                 # Pass feature names explicitly
#                 plt.figure(figsize=(10, 6))
#                 shap.summary_plot(shap_values, st.session_state['X_test_scaled'], 
#                                     feature_names=st.session_state['feature_names'])  # Use feature names from session_state
#                 st.pyplot(plt)


#         # SHAP Analysis for XGBoost (Local Instance)
#         if model_option == "XGBoost":
#             st.subheader("SHAP Local Explanation for a Random Instance")

#             # Create SHAP explainer for the model
#             explainer_shap = shap.Explainer(model)

#             # Select a random instance from X_test_scaled
#             i = np.random.randint(0, st.session_state['X_test_scaled'].shape[0])
#             instance_to_explain = st.session_state['X_test_scaled'][i].reshape(1, -1)

#             # Compute SHAP values for the instance
#             shap_values = explainer_shap(instance_to_explain)

#             # Display the selected instance index
#             st.write(f"**Explaining Instance {i}:**")

#             # Waterfall Plot
#             st.subheader("SHAP Waterfall Plot for the Instance")
#             fig_waterfall = plt.figure(figsize=(10, 6))
#             shap.waterfall_plot(shap_values[0])  # No ax parameter
#             st.pyplot(fig_waterfall)

#             # Force Plot (if needed)
#             st.subheader("SHAP Force Plot for the Instance")
#             force_fig = plt.figure(figsize=(10, 6))
#             shap.force_plot(explainer_shap.expected_value, shap_values.values, instance_to_explain[0], matplotlib=True)
#             st.pyplot(force_fig)

# ########################### SHAP ANALYSIS  ###########

   
    # Assuming you have already loaded your data and defined X_train, y_train, X_test
    # Initialize your model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Fit the model to the training data
    xgb_model.fit(X_train, y_train)

    # Create the SHAP explainer using the model and training data
    explainer = shap.Explainer(xgb_model, X_train)

    # Generate SHAP values for the test set
    explanations = explainer(X_test, check_additivity=False)

    # Streamlit App
    st.title("SHAP Analysis for XGBoost Model")

    # Local SHAP Explanation for a Specific Row
    row_idx = st.number_input("Select the row index to explain", min_value=0, max_value=len(X_test)-1, value=1)

    # Waterfall Plot for a Specific Prediction
    st.subheader(f"SHAP Waterfall Plot for Instance {row_idx}")
    fig_waterfall = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanations[row_idx])  # Waterfall plot for a specific prediction
    st.pyplot(fig_waterfall)

    # Global SHAP Explanation - Bar Plot of Feature Importance
    st.subheader("Global SHAP Feature Importance")
    fig_summary = plt.figure(figsize=(10, 6))
    shap.summary_plot(explanations, plot_type="bar")  # Global feature importance
    st.pyplot(fig_summary)



        
            
else:
    st.warning("Model is not available.")
