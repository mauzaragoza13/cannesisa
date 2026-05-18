import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Cannes Creative Potential Calculator",
    page_icon="🏆",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def contains_any(text, words):
    text = str(text).lower()
    return any(w.lower() in text for w in words)


def count_keywords(text, words):
    text = str(text).lower()
    return sum(1 for w in words if w.lower() in text)


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


# ============================================================
# AUTO-ESTIMATION RULES
# ============================================================

IDEA_KEYWORDS = {
    "public relations stunt": [
        "stunt", "activation", "activación", "intervención", "public intervention",
        "earned media", "earned conversation", "guerrilla", "takeover", "challenge",
        "reto", "récord", "record", "world's biggest", "biggest", "largest",
        "más grande", "evento público", "protest", "demonstration", "movimiento"
    ],
    "experiential idea": [
        "experience", "experiential", "experiencia", "installation", "instalación",
        "immersive", "inmersiva", "interactive", "interactiva", "live", "en vivo",
        "pop-up", "museum", "exhibition", "festival", "showroom", "activation", "activación"
    ],
    "social impact": [
        "social", "community", "comunidad", "education", "educación", "health", "salud",
        "mental health", "sustainability", "sostenibilidad", "sustentabilidad", "climate",
        "clima", "environment", "medio ambiente", "children", "niños", "poverty",
        "pobreza", "inclusion", "inclusión", "diversity", "diversidad", "equality",
        "igualdad", "rights", "derechos", "accessibility", "accesibilidad"
    ],
    "activism": [
        "activism", "activismo", "protest", "protesta", "petition", "petición",
        "justice", "justicia", "human rights", "derechos humanos", "racism", "racismo",
        "gender equality", "igualdad de género", "violence", "violencia", "boycott",
        "discrimination", "discriminación", "manifesto", "manifiesto"
    ],
    "technology-driven idea": [
        "ai", "artificial intelligence", "inteligencia artificial", "machine learning",
        "algorithm", "algoritmo", "sensor", "blockchain", "augmented reality", "realidad aumentada",
        "virtual reality", "realidad virtual", "api", "iot", "automation", "automatización",
        "robot", "facial recognition", "voice recognition", "computer vision", "data model"
    ],
    "utility idea": [
        "tool", "herramienta", "solution", "solución", "service", "servicio", "help",
        "ayuda", "guide", "guía", "map", "mapa", "calculator", "calculadora",
        "tracker", "helpline", "resource", "recurso", "kit", "training", "entrenamiento"
    ],
    "product innovation": [
        "product", "producto", "prototype", "prototipo", "device", "dispositivo",
        "packaging", "empaque", "material", "bottle", "botella", "wearable", "new product",
        "limited edition", "edición limitada", "designed", "diseñado"
    ],
    "documentary": [
        "documentary", "documental", "docuseries", "film series", "real story", "historia real",
        "true story", "interview", "entrevista", "testimony", "testimonio", "portrait", "retrato"
    ],
    "humor": [
        "funny", "humor", "laugh", "risa", "joke", "broma", "comedy", "comedia",
        "parody", "parodia", "prank", "meme", "satire", "sátira", "ridiculous"
    ],
    "absurdity": [
        "absurd", "absurdo", "weird", "raro", "strange", "extraño", "crazy",
        "loco", "unexpected", "inesperado", "bizarre", "surreal", "impossible", "imposible"
    ],
    "innovation": [
        "innovation", "innovación", "world's first", "first ever", "primera vez",
        "breakthrough", "reinvented", "reinventado", "transformed", "transformado",
        "new way", "nueva forma", "never before", "nunca antes"
    ],
    "emotional storytelling": [
        "story", "historia", "life", "vida", "family", "familia", "mother", "madre",
        "father", "padre", "home", "hogar", "dream", "sueño", "love", "amor",
        "memory", "memoria", "human", "humano", "journey", "viaje", "tribute", "homenaje"
    ]
}

SENTIMENT_KEYWORDS = {
    "funny": ["funny", "humor", "laugh", "risa", "joke", "broma", "comedy", "parody", "meme", "satire"],
    "hopeful": ["hope", "esperanza", "future", "futuro", "change", "cambio", "better", "mejor", "progress", "progreso", "support", "apoyo"],
    "inspirational": ["inspire", "inspirar", "hero", "héroe", "courage", "valor", "brave", "valiente", "empower", "empoderar", "celebrate", "celebrar"],
    "sad": ["sad", "triste", "lost", "perdido", "death", "muerte", "grief", "duelo", "abuse", "abuso", "hunger", "hambre", "poverty", "pobreza", "illness", "enfermedad"],
    "tense": ["anger", "enojo", "fight", "lucha", "war", "guerra", "conflict", "conflicto", "crisis", "protest", "protesta", "threat", "amenaza", "risk", "riesgo"],
    "shocking": ["shock", "impactante", "danger", "peligro", "unexpected", "inesperado", "surprise", "sorpresa", "taboo", "scandal", "escándalo", "hidden", "oculto", "exposed"],
    "nostalgic": ["memory", "memoria", "remember", "recordar", "childhood", "infancia", "home", "hogar", "past", "pasado", "heritage", "tradición", "legacy", "legado"]
}

CATEGORY_PROFILES = {
    "film": {
        "emotional": 0.25, "culture": 0.15, "viral": 0.10, "simplicity": 0.20,
        "execution": 0.20, "tech": 0.03, "social": 0.07
    },
    "film craft": {
        "emotional": 0.15, "culture": 0.10, "viral": 0.05, "simplicity": 0.10,
        "execution": 0.45, "tech": 0.05, "social": 0.10
    },
    "public relations": {
        "emotional": 0.08, "culture": 0.22, "viral": 0.35, "simplicity": 0.10,
        "execution": 0.08, "tech": 0.04, "social": 0.13
    },
    "pr": {
        "emotional": 0.08, "culture": 0.22, "viral": 0.35, "simplicity": 0.10,
        "execution": 0.08, "tech": 0.04, "social": 0.13
    },
    "outdoor": {
        "emotional": 0.08, "culture": 0.18, "viral": 0.28, "simplicity": 0.22,
        "execution": 0.14, "tech": 0.03, "social": 0.07
    },
    "innovation": {
        "emotional": 0.05, "culture": 0.12, "viral": 0.08, "simplicity": 0.10,
        "execution": 0.18, "tech": 0.32, "social": 0.15
    },
    "titanium": {
        "emotional": 0.15, "culture": 0.25, "viral": 0.20, "simplicity": 0.10,
        "execution": 0.10, "tech": 0.08, "social": 0.12
    },
    "brand experience": {
        "emotional": 0.10, "culture": 0.15, "viral": 0.22, "simplicity": 0.10,
        "execution": 0.25, "tech": 0.08, "social": 0.10
    },
    "direct": {
        "emotional": 0.08, "culture": 0.14, "viral": 0.15, "simplicity": 0.20,
        "execution": 0.10, "tech": 0.08, "social": 0.25
    },
    "design": {
        "emotional": 0.08, "culture": 0.12, "viral": 0.08, "simplicity": 0.20,
        "execution": 0.32, "tech": 0.08, "social": 0.12
    },
    "media": {
        "emotional": 0.08, "culture": 0.18, "viral": 0.30, "simplicity": 0.12,
        "execution": 0.10, "tech": 0.10, "social": 0.12
    },
    "default": {
        "emotional": 0.14, "culture": 0.18, "viral": 0.18, "simplicity": 0.15,
        "execution": 0.15, "tech": 0.08, "social": 0.12
    }
}


def get_category_profile(category):
    c = str(category).lower()
    for key, profile in CATEGORY_PROFILES.items():
        if key != "default" and key in c:
            return profile
    return CATEGORY_PROFILES["default"]


def auto_estimate_idea_type(text, category):
    full_text = f"{text} {category}".lower()
    scores = {idea: count_keywords(full_text, kws) for idea, kws in IDEA_KEYWORDS.items()}

    c = str(category).lower()
    if "film" in c:
        scores["emotional storytelling"] += 2
        scores["documentary"] += 1
    if "public relations" in c or c.strip() == "pr":
        scores["public relations stunt"] += 2
    if "outdoor" in c:
        scores["public relations stunt"] += 1
        scores["experiential idea"] += 1
    if "experience" in c or "activation" in c:
        scores["experiential idea"] += 2
    if "innovation" in c:
        scores["innovation"] += 2
        scores["technology-driven idea"] += 1
    if "design" in c:
        scores["product innovation"] += 1
    if "health" in c or "sustainable" in c:
        scores["social impact"] += 2

    strong_tech = count_keywords(full_text, IDEA_KEYWORDS["technology-driven idea"])
    if strong_tech == 0:
        scores["technology-driven idea"] = 0

    priority = [
        "activism", "social impact", "public relations stunt", "experiential idea",
        "utility idea", "product innovation", "documentary", "humor", "absurdity",
        "emotional storytelling", "technology-driven idea", "innovation"
    ]

    max_score = max(scores.values())
    if max_score == 0:
        return "emotional storytelling"

    candidates = [k for k, v in scores.items() if v == max_score]
    for p in priority:
        if p in candidates:
            return p
    return candidates[0]


def auto_estimate_sentiment(text, idea_type):
    full_text = str(text).lower()
    scores = {sent: count_keywords(full_text, kws) for sent, kws in SENTIMENT_KEYWORDS.items()}

    if idea_type == "humor":
        scores["funny"] += 3
    if idea_type == "absurdity":
        scores["funny"] += 1
        scores["shocking"] += 2
    if idea_type == "activism":
        scores["tense"] += 2
        scores["hopeful"] += 1
    if idea_type == "social impact":
        scores["hopeful"] += 2
        scores["inspirational"] += 1
    if idea_type == "emotional storytelling":
        scores["nostalgic"] += 1
        scores["inspirational"] += 1
    if idea_type == "documentary":
        scores["sad"] += 1
        scores["inspirational"] += 1
    if idea_type == "public relations stunt":
        scores["shocking"] += 1

    priority = ["shocking", "funny", "tense", "hopeful", "nostalgic", "sad", "inspirational"]
    max_score = max(scores.values())
    if max_score == 0:
        if idea_type == "public relations stunt":
            return "shocking"
        if idea_type == "social impact":
            return "hopeful"
        if idea_type == "documentary":
            return "sad"
        if idea_type == "emotional storytelling":
            return "nostalgic"
        return "hopeful"

    candidates = [k for k, v in scores.items() if v == max_score]
    for p in priority:
        if p in candidates:
            return p
    return candidates[0]


def auto_estimate_scores(project_name, description, category, idea_type, sentiment):
    text = f"{project_name} {description} {category} {idea_type} {sentiment}".lower()

    social_words = IDEA_KEYWORDS["social impact"] + IDEA_KEYWORDS["activism"]
    viral_words = IDEA_KEYWORDS["public relations stunt"] + ["viral", "share", "conversation", "conversación", "famoso"]
    tech_words = IDEA_KEYWORDS["technology-driven idea"]
    complexity_words = [
        "installation", "instalación", "event", "evento", "live", "en vivo", "platform", "plataforma",
        "app", "technology", "tecnología", "film", "documentary", "documental", "production", "producción"
    ]
    simplicity_words = ["simple", "simplicity", "clarity", "claro", "directo", "fácil", "easy", "one idea", "memorable"]

    social_count = count_keywords(text, social_words)
    viral_count = count_keywords(text, viral_words)
    tech_count = count_keywords(text, tech_words)
    complexity_count = count_keywords(text, complexity_words)
    simplicity_count = count_keywords(text, simplicity_words)

    cultural_relevance = 5 + min(4, social_count * 1.2)
    viral_potential = 5 + min(4, viral_count * 1.2)
    simplicity_of_insight = 7 + min(2, simplicity_count * 0.7)
    execution_complexity = 5 + min(4, complexity_count * 0.8)

    tech_component_ai = 1 if tech_count > 0 or idea_type == "technology-driven idea" else 0
    social_impact_ai = 1 if social_count > 0 or idea_type in ["social impact", "activism"] else 0

    if idea_type == "public relations stunt":
        viral_potential += 1.5
        execution_complexity += 0.5
    if idea_type == "experiential idea":
        viral_potential += 0.8
        execution_complexity += 1.2
    if idea_type == "emotional storytelling":
        cultural_relevance += 0.8
        simplicity_of_insight += 0.5
    if idea_type == "technology-driven idea":
        execution_complexity += 2
    if idea_type == "utility idea":
        simplicity_of_insight += 0.5
    if idea_type == "product innovation":
        execution_complexity += 1
    if sentiment in ["shocking", "tense", "funny"]:
        viral_potential += 1
    if sentiment in ["hopeful", "nostalgic", "sad", "inspirational"]:
        cultural_relevance += 0.5

    c = str(category).lower()
    if "public relations" in c or c.strip() == "pr":
        viral_potential += 1
    if "film" in c:
        simplicity_of_insight += 0.5
    if "innovation" in c:
        execution_complexity += 1
    if "outdoor" in c:
        viral_potential += 0.8
        simplicity_of_insight += 0.5

    return {
        "cultural_relevance": round(clamp(cultural_relevance, 1, 10), 1),
        "viral_potential": round(clamp(viral_potential, 1, 10), 1),
        "simplicity_of_insight": round(clamp(simplicity_of_insight, 1, 10), 1),
        "execution_complexity": round(clamp(execution_complexity, 1, 10), 1),
        "tech_component_ai": int(tech_component_ai),
        "social_impact_ai": int(social_impact_ai)
    }


def auto_estimate_jury_fit(category, idea_type, sentiment, scores):
    profile = get_category_profile(category)

    emotional_signal = 0.4
    if idea_type == "emotional storytelling":
        emotional_signal += 0.35
    if sentiment in ["hopeful", "nostalgic", "sad", "inspirational"]:
        emotional_signal += 0.25
    emotional_signal = clamp(emotional_signal, 0, 1)

    culture = scores["cultural_relevance"] / 10
    viral = scores["viral_potential"] / 10
    simplicity = scores["simplicity_of_insight"] / 10
    execution = scores["execution_complexity"] / 10
    tech = scores["tech_component_ai"]
    social = scores["social_impact_ai"]

    if idea_type == "public relations stunt":
        viral = clamp(viral + 0.20, 0, 1)
    if idea_type in ["utility idea", "product innovation"]:
        execution = clamp(execution + 0.10, 0, 1)
    if idea_type in ["social impact", "activism"]:
        social = 1
    if idea_type == "technology-driven idea":
        tech = 1

    raw = (
        profile["emotional"] * emotional_signal +
        profile["culture"] * culture +
        profile["viral"] * viral +
        profile["simplicity"] * simplicity +
        profile["execution"] * execution +
        profile["tech"] * tech +
        profile["social"] * social
    )

    return round(clamp(raw * 100, 0, 100), 1)


def auto_estimate_all(project_name, description, category):
    text = f"{project_name} {description}"
    idea_type = auto_estimate_idea_type(text, category)
    sentiment = auto_estimate_sentiment(text, idea_type)
    scores = auto_estimate_scores(project_name, description, category, idea_type, sentiment)
    jury_fit_score = auto_estimate_jury_fit(category, idea_type, sentiment, scores)
    jury_fit_status = fit_status_from_score(jury_fit_score)

    return {
        "idea_type_ai": idea_type,
        "sentiment_ai": sentiment,
        **scores,
        "jury_fit_score": jury_fit_score,
        "jury_fit_status": jury_fit_status,
        "classification_confidence": 0.75
    }


# ============================================================
# PREDICTION AND EXPLANATION
# ============================================================

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


def explain_campaign(inputs, predicted_score, prob_high_award):
    strengths = []
    risks = []

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


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🏆 Cannes Creative Potential Calculator")
st.caption("Modelo exploratorio para estimar potencial Cannes a partir de variables creativas, categoría y afinidad con jurado. No usa marca ni agencia como predictor.")

with st.sidebar:
    st.header("Configuración")
    st.info("Primero escribe la descripción y usa Autoestimar. Después ajusta manualmente si lo necesitas.")
    show_importance = st.checkbox("Mostrar importancia de variables", value=True)

category_options = training_categories.get("category_norm", []) or [
    "Film", "Outdoor", "Titanium", "Innovation", "Public Relations", "Brand Experience & Activation", "Media", "Design", "Direct"
]
subcategory_options = training_categories.get("subcategory", []) or ["Unknown"]
idea_type_options = training_categories.get("idea_type_ai", []) or list(IDEA_KEYWORDS.keys())
sentiment_options = training_categories.get("sentiment_ai", []) or list(SENTIMENT_KEYWORDS.keys())

# Session state defaults
DEFAULTS = {
    "idea_type_ai": "emotional storytelling",
    "sentiment_ai": "hopeful",
    "cultural_relevance": 7.0,
    "viral_potential": 7.0,
    "simplicity_of_insight": 8.0,
    "execution_complexity": 6.0,
    "tech_component_ai": 0,
    "social_impact_ai": 0,
    "jury_fit_score": 65.0,
    "classification_confidence": 0.80
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.subheader("Evaluar una idea")

col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Nombre del proyecto", "La Ola más grande del mundo")
    project_description = st.text_area(
        "Descripción de la idea",
        "Describe brevemente qué sucede, dónde vive la campaña, qué problema resuelve y por qué sería relevante.",
        height=140
    )
    category_norm = st.selectbox("Categoría Cannes", category_options)
    subcategory = st.selectbox("Subcategoría", subcategory_options)

    if st.button("Autoestimar variables", type="secondary"):
        auto = auto_estimate_all(project_name, project_description, category_norm)
        for key, value in auto.items():
            st.session_state[key] = value
        st.success("Variables autoestimadas. Revísalas y ajusta manualmente si hace falta.")

with col2:
    idea_index = idea_type_options.index(st.session_state["idea_type_ai"]) if st.session_state["idea_type_ai"] in idea_type_options else 0
    sentiment_index = sentiment_options.index(st.session_state["sentiment_ai"]) if st.session_state["sentiment_ai"] in sentiment_options else 0

    idea_type_ai = st.selectbox("Tipo de idea", idea_type_options, index=idea_index, key="idea_type_ai_select")
    sentiment_ai = st.selectbox("Sentimiento principal", sentiment_options, index=sentiment_index, key="sentiment_ai_select")

    # Sync selectbox values into session state for scoring
    st.session_state["idea_type_ai"] = idea_type_ai
    st.session_state["sentiment_ai"] = sentiment_ai

st.divider()
st.markdown("### Variables creativas autoestimadas / editables")

col3, col4 = st.columns(2)
with col3:
    cultural_relevance = st.slider("Relevancia cultural", 1.0, 10.0, float(st.session_state["cultural_relevance"]), 0.5, key="cultural_relevance_slider")
    viral_potential = st.slider("Potencial viral / PR", 1.0, 10.0, float(st.session_state["viral_potential"]), 0.5, key="viral_potential_slider")
    simplicity_of_insight = st.slider("Simplicidad del insight", 1.0, 10.0, float(st.session_state["simplicity_of_insight"]), 0.5, key="simplicity_slider")
    execution_complexity = st.slider("Complejidad de ejecución", 1.0, 10.0, float(st.session_state["execution_complexity"]), 0.5, key="execution_slider")

with col4:
    tech_component_ai = st.radio(
        "Componente tecnológico",
        [0, 1],
        index=int(st.session_state["tech_component_ai"]),
        format_func=lambda x: "Sí" if x == 1 else "No",
        horizontal=True,
        key="tech_radio"
    )
    social_impact_ai = st.radio(
        "Impacto social",
        [0, 1],
        index=int(st.session_state["social_impact_ai"]),
        format_func=lambda x: "Sí" if x == 1 else "No",
        horizontal=True,
        key="social_radio"
    )

    auto_jury_fit = auto_estimate_jury_fit(
        category_norm,
        idea_type_ai,
        sentiment_ai,
        {
            "cultural_relevance": cultural_relevance,
            "viral_potential": viral_potential,
            "simplicity_of_insight": simplicity_of_insight,
            "execution_complexity": execution_complexity,
            "tech_component_ai": tech_component_ai,
            "social_impact_ai": social_impact_ai,
        }
    )

    if st.button("Recalcular Jury Fit con variables actuales"):
        st.session_state["jury_fit_score"] = auto_jury_fit

    jury_fit_score = st.slider("Jury Fit Score", 0.0, 100.0, float(st.session_state["jury_fit_score"]), 1.0, key="jury_slider")
    jury_fit_status = fit_status_from_score(jury_fit_score)
    st.write(f"Afinidad: **{jury_fit_status}**")

classification_confidence = st.slider("Confianza de clasificación", 0.0, 1.0, float(st.session_state["classification_confidence"]), 0.05)

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

    result_df = pd.DataFrame([{
        "project_name": project_name,
        "project_description": project_description,
        **input_row,
        "predicted_cannes_score": predicted_score,
        "prob_high_award": prob_high_award,
        "prob_high_award_percent": round(prob_high_award * 100, 1),
        "potential_label": label,
        "strengths": "; ".join(strengths),
        "risks": "; ".join(risks)
    }])

    st.download_button(
        label="Descargar resultado CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="cannes_prediction_result.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("Evaluación masiva")
st.write("Sube un CSV o Excel con las columnas de entrada para evaluar varias ideas.")

bulk_file = st.file_uploader("Subir archivo de ideas", type=["csv", "xlsx"])
if bulk_file is not None:
    if bulk_file.name.endswith(".csv"):
        ideas_df = pd.read_csv(bulk_file)
    else:
        ideas_df = pd.read_excel(bulk_file)

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
        st.download_button(
            label="Descargar evaluación masiva CSV",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="cannes_bulk_predictions.csv",
            mime="text/csv"
        )

if show_importance and not importance_df.empty:
    st.divider()
    st.subheader("Top variables del modelo")
    st.dataframe(importance_df.head(20), use_container_width=True)
    st.bar_chart(importance_df.head(15).set_index("feature")["importance"])

st.divider()
st.caption("Modelo exploratorio desarrollado para ISA. No debe interpretarse como garantía de premio; sirve como herramienta de priorización creativa.")
