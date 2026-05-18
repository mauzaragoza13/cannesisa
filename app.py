import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Cannes Creative Potential Calculator",
    page_icon="🏆",
    layout="wide"
)

@st.cache_resource
def load_model_bundle():
    return joblib.load("cannes_model_bundle.pkl")

bundle = load_model_bundle()
rf_regressor_final = bundle["rf_regressor_final"]
rf_classifier_final = bundle["rf_classifier_final"]
features = bundle["features"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
importance_df = bundle.get("importance_df", pd.DataFrame())
training_categories = bundle.get("training_categories", {})

def score_label(score):
    if score >= 90:
        return "Potencial Gold / Grand Prix"
    elif score >= 80:
        return "Potencial metal"
    elif score >= 70:
        return "Potencial shortlist / bronze"
    return "Potencial bajo"

def fit_status_from_score(score):
    if score >= 85:
        return "alta afinidad"
    elif score >= 70:
        return "buena afinidad"
    elif score >= 50:
        return "afinidad media"
    return "baja afinidad"

def explain_campaign(inputs, predicted_score, prob_high_award):
    strengths, risks = [], []
    if inputs["viral_potential"] >= 8:
        strengths.append("Alto potencial viral / conversación")
    elif inputs["viral_potential"] <= 5:
        risks.append("Potencial viral bajo o moderado")
    if inputs["cultural_relevance"] >= 8:
        strengths.append("Alta relevancia cultural")
    elif inputs["cultural_relevance"] <= 5:
        risks.append("Relevancia cultural baja o moderada")
    if inputs["simplicity_of_insight"] >= 8:
        strengths.append("Insight simple y claro")
    elif inputs["simplicity_of_insight"] <= 5:
        risks.append("Insight complejo o poco claro")
    if inputs["social_impact_ai"] == 1:
        strengths.append("Componente de impacto social")
    if inputs["tech_component_ai"] == 1:
        strengths.append("Componente tecnológico")
    if inputs["jury_fit_score"] >= 75:
        strengths.append("Buena afinidad con jurado/categoría")
    elif inputs["jury_fit_score"] < 50:
        risks.append("Baja afinidad con jurado/categoría")
    if inputs["execution_complexity"] >= 8:
        risks.append("Alta complejidad de ejecución")
    if predicted_score >= 90:
        strengths.append("Score estimado en rango alto de Cannes")
    elif predicted_score < 70:
        risks.append("Score estimado por debajo del rango competitivo")
    if prob_high_award >= 0.70:
        strengths.append("Alta probabilidad estimada de premio alto")
    elif prob_high_award < 0.40:
        risks.append("Probabilidad estimada de premio alto limitada")
    return strengths, risks

def predict_campaign_potential(input_row):
    new_data = pd.DataFrame([input_row])
    for col in numeric_features:
        new_data[col] = pd.to_numeric(new_data[col], errors="coerce").fillna(0)
    for col in categorical_features:
        new_data[col] = new_data[col].fillna("Unknown").astype(str)
    new_data = new_data[features]
    predicted_score = rf_regressor_final.predict(new_data)[0]
    prob_high_award = rf_classifier_final.predict_proba(new_data)[0, 1]
    return round(float(predicted_score), 1), round(float(prob_high_award), 3)

st.title("🏆 Cannes Creative Potential Calculator")
st.caption("Modelo exploratorio para estimar potencial Cannes a partir de variables creativas, categoría y afinidad con jurado. No usa marca ni agencia como predictor.")

with st.sidebar:
    st.header("Configuración")
    st.info("Modelo exploratorio. Úsalo para priorizar ideas, no como garantía de premio.")
    show_importance = st.checkbox("Mostrar importancia de variables", value=True)

st.subheader("Evaluar una idea")

category_options = training_categories.get("category_norm", []) or ["Film", "Outdoor", "Titanium", "Innovation", "Public Relations", "Brand Experience & Activation"]
subcategory_options = training_categories.get("subcategory", []) or ["Unknown"]
idea_type_options = training_categories.get("idea_type_ai", []) or [
    "emotional storytelling", "public relations stunt", "social impact", "documentary",
    "experiential idea", "innovation", "utility idea", "product innovation",
    "technology-driven idea", "absurdity", "humor", "activism"
]
sentiment_options = training_categories.get("sentiment_ai", []) or ["nostalgic", "hopeful", "shocking", "sad", "inspirational", "tense", "funny"]

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Nombre del proyecto", "Nueva idea ISA")
    category_norm = st.selectbox("Categoría Cannes", category_options)
    subcategory = st.selectbox("Subcategoría", subcategory_options)
    idea_type_ai = st.selectbox("Tipo de idea", idea_type_options)
    sentiment_ai = st.selectbox("Sentimiento principal", sentiment_options)
with col2:
    cultural_relevance = st.slider("Relevancia cultural", 1.0, 10.0, 7.0, 0.5)
    viral_potential = st.slider("Potencial viral / PR", 1.0, 10.0, 7.0, 0.5)
    simplicity_of_insight = st.slider("Simplicidad del insight", 1.0, 10.0, 8.0, 0.5)
    execution_complexity = st.slider("Complejidad de ejecución", 1.0, 10.0, 6.0, 0.5)

col3, col4, col5 = st.columns(3)
with col3:
    tech_component_ai = st.radio("Componente tecnológico", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
with col4:
    social_impact_ai = st.radio("Impacto social", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
with col5:
    jury_fit_score = st.slider("Jury Fit Score", 0.0, 100.0, 65.0, 1.0)
    jury_fit_status = fit_status_from_score(jury_fit_score)
    st.write(f"Afinidad: **{jury_fit_status}**")
classification_confidence = st.slider("Confianza de clasificación", 0.0, 1.0, 0.80, 0.05)

input_row = {
    "cultural_relevance": cultural_relevance,
    "viral_potential": viral_potential,
    "simplicity_of_insight": simplicity_of_insight,
    "execution_complexity": execution_complexity,
    "tech_component_ai": tech_component_ai,
    "social_impact_ai": social_impact_ai,
    "jury_fit_score": jury_fit_score,
    "classification_confidence": classification_confidence,
    "category_norm": category_norm,
    "subcategory": subcategory,
    "idea_type_ai": idea_type_ai,
    "sentiment_ai": sentiment_ai,
    "jury_fit_status": jury_fit_status
}

if st.button("Calcular potencial Cannes", type="primary"):
    predicted_score, prob_high_award = predict_campaign_potential(input_row)
    label = score_label(predicted_score)
    strengths, risks = explain_campaign(input_row, predicted_score, prob_high_award)
    st.divider()
    st.subheader(project_name)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Cannes Score estimado", f"{predicted_score}/100")
    metric2.metric("Probabilidad High Award", f"{prob_high_award * 100:.1f}%")
    metric3.metric("Etiqueta", label)
    st.markdown("### Lectura estratégica")
    if strengths:
        st.markdown("**Fortalezas**")
        for item in strengths:
            st.write(f"✅ {item}")
    if risks:
        st.markdown("**Riesgos / áreas a mejorar**")
        for item in risks:
            st.write(f"⚠️ {item}")
    result_df = pd.DataFrame([{ "project_name": project_name, **input_row,
        "predicted_cannes_score": predicted_score,
        "prob_high_award": prob_high_award,
        "prob_high_award_percent": round(prob_high_award * 100, 1),
        "potential_label": label,
        "strengths": "; ".join(strengths),
        "risks": "; ".join(risks)
    }])
    st.download_button("Descargar resultado CSV", data=result_df.to_csv(index=False).encode("utf-8"), file_name="cannes_prediction_result.csv", mime="text/csv")

st.divider()
st.subheader("Evaluación masiva")
st.write("Sube un CSV o Excel con las mismas columnas de entrada para evaluar varias ideas.")
bulk_file = st.file_uploader("Subir archivo de ideas", type=["csv", "xlsx"])
if bulk_file is not None:
    ideas_df = pd.read_csv(bulk_file) if bulk_file.name.endswith(".csv") else pd.read_excel(bulk_file)
    missing = [c for c in features if c not in ideas_df.columns]
    if missing:
        st.error(f"Faltan columnas: {missing}")
    else:
        predictions = []
        for _, row in ideas_df.iterrows():
            row_input = {c: row[c] for c in features}
            pred_score, pred_prob = predict_campaign_potential(row_input)
            predictions.append({
                "predicted_cannes_score": pred_score,
                "prob_high_award": pred_prob,
                "prob_high_award_percent": round(pred_prob * 100, 1),
                "potential_label": score_label(pred_score)
            })
        pred_df = pd.concat([ideas_df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
        st.dataframe(pred_df, use_container_width=True)
        st.download_button("Descargar evaluación masiva CSV", data=pred_df.to_csv(index=False).encode("utf-8"), file_name="cannes_bulk_predictions.csv", mime="text/csv")

if show_importance and not importance_df.empty:
    st.divider()
    st.subheader("Top variables del modelo")
    st.dataframe(importance_df.head(20), use_container_width=True)
    st.bar_chart(importance_df.head(15).set_index("feature")["importance"])

st.divider()
st.caption("Modelo exploratorio desarrollado para ISA. No debe interpretarse como garantía de premio; sirve como herramienta de priorización creativa.")
