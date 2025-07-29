import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import lime
import lime.lime_tabular
import streamlit.components.v1 as components

# === Load DEME model ===
modelo_dict = joblib.load("models/DEME_Model.pkl")
modelos = modelo_dict["modelos"]
scaler = modelo_dict["scaler"]
variables = modelo_dict["variables"]

# === Prediction functions ===
def obtener_predicciones_proba(X_input):
    predicciones = {}
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_input)[:, 1]
        predicciones[nombre] = proba
    return predicciones

def calcular_pesos_dinamicos_por_muestra(predicciones):
    pesos = {}
    total = sum([proba[0] for proba in predicciones.values()])
    for nombre, proba in predicciones.items():
        pesos[nombre] = proba[0] / total if total > 0 else 1 / len(predicciones)
    return pesos

def ensamblar_dinamico(predicciones, pesos):
    n = len(next(iter(predicciones.values())))
    resultado = np.zeros(n)
    for nombre, proba in predicciones.items():
        resultado += pesos[nombre] * proba
    return (resultado >= 0.5).astype(int)

# === Dictionaries ===
app_mode_dict = {
    1: "1st phase - general contingent", 
    2: "Ordinance No. 612/93", 
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses", 
    10: "Ordinance No. 854-B/99", 
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)", 
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent", 
    26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)", 
    39: "Over 23 years old", 
    42: "Transfer",
    43: "Change of course", 
    44: "Technological specialization diploma holders", 
    51: "Change of institution/course",
    53: "Short cycle diploma holders", 
    57: "Change of institution/course (International)"
}

course_dict = {
    33: "Biofuel Production Technologies", 
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)", 
    9003: "Agronomy", 
    9070: "Communication Design",
    9085: "Veterinary Nursing", 
    9119: "Informatics Engineering", 
    9130: "Equinculture", 
    9147: "Management",
    9238: "Social Service", 
    9254: "Tourism", 
    9500: "Nursing", 
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management", 
    9773: "Journalism and Communication", 
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

occupation_dict = {
    0: "Student", 1: "Legislative/Executive Rep/Managers", 2: "Intellectual/Scientific Activities",
    3: "Intermediate Technicians", 4: "Administrative staff", 5: "Personal Services and Sellers",
    6: "Farmers/Fishermen", 7: "Industry/Construction Workers", 8: "Machine Operators",
    9: "Unskilled Workers", 10: "Armed Forces", 90: "Other Situation", 99: "(blank)",
    122: "Health professionals", 123: "Teachers", 125: "ICT Specialists", 131: "Science/Engineering Technicians",
    132: "Health Technicians", 134: "Legal/Social/Sports Technicians", 141: "Office workers",
    143: "Accounting/Finance Operators", 144: "Other Admin Support", 151: "Personal service workers",
    152: "Sellers", 153: "Personal care workers", 171: "Skilled construction workers",
    173: "Craftsmen/Artisans", 175: "Food/Clothing Workers", 191: "Cleaning workers",
    192: "Unskilled Agricultural Workers", 193: "Unskilled Industry Workers", 194: "Meal prep assistants",
    101: "Armed Forces Officers", 102: "Armed Forces Sergeants", 103: "Other Armed Forces personnel",
    112: "Admin/Commercial Directors", 114: "Service Directors", 121: "Science/Math/Eng Specialists",
    124: "Finance/Admin Specialists", 135: "ICT Technicians", 154: "Security Personnel",
    161: "Skilled Agricultural Workers", 163: "Subsistence Farmers", 172: "Metalworkers",
    174: "Electricians/Electronics", 181: "Plant Operators", 182: "Assembly Workers",
    183: "Vehicle Drivers", 195: "Street Vendors"
}

# === Ensemble explanation ===
def mostrar_explicacion_modelos(predicciones):
    st.subheader("Model Ensemble Explanation")
    promedio_por_modelo = {nombre: round(np.mean(pred) * 100, 2) for nombre, pred in predicciones.items()}
    df_modelos = pd.DataFrame(list(promedio_por_modelo.items()), columns=["Model", "Decision Strength (%)"])
    mejor_modelo = df_modelos.loc[df_modelos["Decision Strength (%)"].idxmax()]
    st.dataframe(df_modelos.sort_values(by="Decision Strength (%)", ascending=False), use_container_width=True)
    st.success(f"Most influential model: {mejor_modelo['Model']} with {mejor_modelo['Decision Strength (%)']}%")

# === Streamlit Configuration ===
st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓", layout="wide")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose input mode", ["Student Form", "Student with File"])

if option == "Student Form":
    st.title("Anonymous Form for Students")
    with st.form("anonymous_form"):
        app_mode = st.selectbox("How did you enter university?", options=list(app_mode_dict.keys()), format_func=lambda x: app_mode_dict[x])
        course = st.selectbox("What is your course?", options=list(course_dict.keys()), format_func=lambda x: course_dict[x])
        mother_occ = st.selectbox("Mother's occupation?", options=list(occupation_dict.keys()), format_func=lambda x: occupation_dict[x])
        father_occ = st.selectbox("Father's occupation?", options=list(occupation_dict.keys()), format_func=lambda x: occupation_dict[x])
        adm_grade = st.slider("Admission grade", 0.0, 200.0, 150.0)
        scholarship = st.radio("Do you have a scholarship?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        age_enroll = st.slider("Age at enrollment", 15, 40, 18)
        prev_grade = st.slider("Previous qualification grade", 0.0, 20.0, 12.0)
        grade_1sem = st.slider("1st semester average grade", 0.0, 20.0, 14.0)
        grade_2sem = st.slider("2nd semester average grade", 0.0, 20.0, 14.0)
        approved_1sem = st.slider("Approved courses in 1st semester", 0, 20, 5)
        approved_2sem = st.slider("Approved courses in 2nd semester", 0, 20, 5)
        eval_1sem = st.slider("Evaluations in 1st semester", 0, 20, 6)
        eval_2sem = st.slider("Evaluations in 2nd semester", 0, 20, 6)
        enviar = st.form_submit_button("Submit and Predict")

    if enviar:
        tuition_up = 0
        entrada = pd.DataFrame([[app_mode, course, mother_occ, father_occ, adm_grade,
                                 tuition_up, scholarship, age_enroll, prev_grade,
                                 grade_1sem, grade_2sem, approved_1sem, approved_2sem,
                                 eval_1sem, eval_2sem]], columns=variables)
        entrada_scaled = scaler.transform(entrada)
        predicciones = obtener_predicciones_proba(entrada_scaled)

        pesos_dinamicos = calcular_pesos_dinamicos_por_muestra(predicciones)
        resultado = ensamblar_dinamico(predicciones, pesos_dinamicos)

        estado = "Graduate" if resultado[0] == 1 else "Dropout"
        color = "green" if resultado[0] == 1 else "red"
        decision_strength = sum(
            pesos_dinamicos[nombre] * predicciones[nombre][0]
            for nombre in predicciones
        ) * 100

        st.markdown(f"### Predicted Status: <span style='color:{color}'>{estado}</span>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=decision_strength,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Graduation Likelihood (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        mostrar_explicacion_modelos(predicciones)

        st.subheader("LIME Explanation of the Most Influential Model")

        datos_entrenamiento = pd.read_csv("data.csv", sep=";")
        X_train_scaled = scaler.transform(datos_entrenamiento[variables])

        promedio_por_modelo = {nombre: np.mean(pred) for nombre, pred in predicciones.items()}
        mejor_modelo_nombre = max(promedio_por_modelo, key=promedio_por_modelo.get)
        modelo_seleccionado = modelos[mejor_modelo_nombre]

        st.info(f"Most influential model: **{mejor_modelo_nombre}**")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=variables,
            class_names=["Dropout", "Graduate"],
            mode="classification"
        )

        exp = explainer.explain_instance(
            data_row=entrada_scaled[0],
            predict_fn=modelo_seleccionado.predict_proba
        )
        exp.save_to_file("lime_explanation.html")
        with open("lime_explanation.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600, scrolling=True)

elif option == "Student with File":
    st.title("Batch Prediction from File")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=";")
            X_scaled = scaler.transform(df[variables])
            predicciones = obtener_predicciones_proba(X_scaled)
            pesos_por_muestra = [calcular_pesos_dinamicos_por_muestra({k: [v[i]] for k, v in predicciones.items()}) for i in range(len(df))]
            resultados = [ensamblar_dinamico({k: np.array([v[i]]) for k, v in predicciones.items()}, pesos) for i, pesos in enumerate(pesos_por_muestra)]

            df_resultados = df.copy()
            df_resultados["Prediction"] = ["Graduate" if r == 1 else "Dropout" for r in resultados]

            st.dataframe(df_resultados[["Prediction"] + variables], use_container_width=True)
            st.success("Predictions completed successfully.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
