import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(
    page_title="Cannes Creative Potential Calculator",
    page_icon="🦁",
    layout="wide"
)

# ============================================================
# CARGA DE MODELOS Y ARCHIVOS
# ============================================================

@st.cache_resource
def load_model_bundle():
    return joblib.load("cannes_model_bundle.pkl")


@st.cache_resource
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo semántico. Se usarán reglas simples. Detalle: {e}")
        return None


@st.cache_data
def load_jury_profiles():
    try:
        jury_df = pd.read_csv("jury_profiles.csv")
        jury_df.columns = jury_df.columns.str.strip()
        return jury_df
    except Exception:
        return pd.DataFrame()


bundle = load_model_bundle()

rf_regressor_final = bundle["rf_regressor_final"]
rf_classifier_final = bundle["rf_classifier_final"]
features = bundle["features"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
importance_df = bundle.get("importance_df", pd.DataFrame())
training_categories = bundle.get("training_categories", {})

embedding_model = load_embedding_model()
jury_profiles_df = load_jury_profiles()


# ============================================================
# FUNCIONES BASE
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


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


def normalize_key(x):
    x = str(x).lower().strip()
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
        "&": "and"
    }

    for a, b in replacements.items():
        x = x.replace(a, b)

    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def get_jury_profile(category_norm):
    if jury_profiles_df.empty:
        return None

    category_key = normalize_key(category_norm)

    temp = jury_profiles_df.copy()
    temp["category_key"] = temp["category_norm"].apply(normalize_key)

    exact_match = temp[temp["category_key"] == category_key]

    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()

    partial_match = temp[
        temp["category_key"].apply(lambda x: x in category_key or category_key in x)
    ]

    if not partial_match.empty:
        return partial_match.iloc[0].to_dict()

    return None


def has_enough_creative_information(project_name, description):
    text = f"{project_name} {description}".lower().strip()
    words = [w for w in text.split() if len(w) > 2]

    creative_signal_words = [
        "campaña", "activación", "instalación", "experiencia", "evento",
        "documental", "historia", "personas", "comunidad", "social",
        "cultura", "viral", "medio", "medios", "pr", "mundo", "récord",
        "record", "reto", "challenge", "acción", "intervención",
        "metro", "aeropuerto", "plaza", "ciudad", "mundial", "méxico",
        "tecnología", "app", "sensor", "ai", "inteligencia", "datos",
        "impacto", "sustentabilidad", "sostenibilidad", "salud",
        "educación", "derechos", "marca", "público", "audiencia"
    ]

    signal_count = sum(1 for w in creative_signal_words if w in text)

    if len(words) < 12 or signal_count == 0:
        return False

    return True


# ============================================================
# PROTOTIPOS SEMÁNTICOS PARA IA ABIERTA
# ============================================================

IDEA_PROTOTYPES = {
    "emotional storytelling": (
        "A campaign based on human truth, emotion, personal stories, family, memory, "
        "hope, identity, dreams, pride or a moving narrative."
    ),
    "public relations stunt": (
        "A public relations stunt, activation or public intervention designed to generate "
        "earned media, public conversation, virality or cultural attention."
    ),
    "social impact": (
        "A campaign with social impact addressing community, sustainability, education, "
        "health, inclusion, equality, rights, climate or social change."
    ),
    "activism": (
        "An activism campaign focused on protest, justice, rights, discrimination, gender "
        "equality, climate action, violence, petitions or public movements."
    ),
    "technology-driven idea": (
        "A technology-driven idea using artificial intelligence, data, algorithms, sensors, "
        "automation, robotics, augmented reality, virtual reality or software."
    ),
    "experiential idea": (
        "An experiential campaign with an installation, live experience, immersive activation, "
        "event, pop-up, exhibition or interactive public experience."
    ),
    "utility idea": (
        "A useful tool, service, guide, platform, kit, map or practical solution that helps "
        "people solve a problem."
    ),
    "product innovation": (
        "A product innovation, prototype, new material, packaging, device, limited edition "
        "product or redesigned object."
    ),
    "documentary": (
        "A documentary, real story, testimony, interview, portrait, docuseries or case film "
        "centered on real people or true events."
    ),
    "humor": (
        "A funny campaign based on comedy, parody, jokes, satire, memes, pranks or entertaining humor."
    ),
    "absurdity": (
        "An absurd, bizarre, surreal, weird, unexpected or impossible idea that creates attention."
    ),
    "innovation": (
        "A breakthrough innovation or new way of doing something, first-ever idea, reinvention, "
        "transformation or pioneering creative solution."
    )
}


SENTIMENT_PROTOTYPES = {
    "funny": "The idea feels funny, comedic, playful, humorous, satirical or entertaining.",
    "hopeful": "The idea feels hopeful, optimistic, positive, future-facing and uplifting.",
    "inspirational": "The idea feels inspirational, empowering, heroic, brave or celebratory.",
    "sad": "The idea feels sad, emotional, painful, tragic, about loss, grief, poverty or suffering.",
    "tense": "The idea feels tense, urgent, confrontational, risky, conflict-driven or protest-like.",
    "shocking": "The idea feels shocking, surprising, provocative, unexpected, taboo or revealing.",
    "nostalgic": "The idea feels nostalgic, tied to memory, childhood, heritage, home, tradition or legacy."
}


# ============================================================
# KEYWORDS PARA RESPALDO DE CLASIFICACIÓN
# ============================================================

IDEA_KEYWORDS = {
    "public relations stunt": [
        "stunt", "activation", "activación", "intervención", "earned media",
        "guerrilla", "takeover", "challenge", "reto", "récord", "record",
        "world's biggest", "biggest", "largest", "más grande", "evento público",
        "protest", "demonstration", "movimiento", "ola", "metro", "aeropuerto"
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
        "algorithm", "algoritmo", "sensor", "blockchain", "augmented reality",
        "realidad aumentada", "virtual reality", "realidad virtual", "api", "iot",
        "automation", "automatización", "robot", "facial recognition",
        "voice recognition", "computer vision", "data model"
    ],
    "utility idea": [
        "tool", "herramienta", "solution", "solución", "service", "servicio", "help",
        "ayuda", "guide", "guía", "map", "mapa", "calculator", "calculadora",
        "tracker", "helpline", "resource", "recurso", "kit", "training", "entrenamiento"
    ],
    "product innovation": [
        "product", "producto", "prototype", "prototipo", "device", "dispositivo",
        "packaging", "empaque", "material", "bottle", "botella", "wearable",
        "new product", "limited edition", "edición limitada", "designed", "diseñado"
    ],
    "documentary": [
        "documentary", "documental", "docuseries", "film series", "real story",
        "historia real", "true story", "interview", "entrevista", "testimony",
        "testimonio", "portrait", "retrato"
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
        "memory", "memoria", "human", "humano", "journey", "viaje", "tribute", "homenaje",
        "orgullo", "aspiración", "país", "méxico"
    ]
}


SENTIMENT_KEYWORDS = {
    "funny": [
        "funny", "humor", "laugh", "risa", "joke", "broma", "comedy", "parody", "meme", "satire"
    ],
    "hopeful": [
        "hope", "esperanza", "future", "futuro", "change", "cambio", "better", "mejor",
        "progress", "progreso", "support", "apoyo", "aspiración", "mundial"
    ],
    "inspirational": [
        "inspire", "inspirar", "hero", "héroe", "courage", "valor", "brave", "valiente",
        "empower", "empoderar", "celebrate", "celebrar", "orgullo"
    ],
    "sad": [
        "sad", "triste", "lost", "perdido", "death", "muerte", "grief", "duelo",
        "abuse", "abuso", "hunger", "hambre", "poverty", "pobreza", "illness", "enfermedad"
    ],
    "tense": [
        "anger", "enojo", "fight", "lucha", "war", "guerra", "conflict", "conflicto",
        "crisis", "protest", "protesta", "threat", "amenaza", "risk", "riesgo"
    ],
    "shocking": [
        "shock", "impactante", "danger", "peligro", "unexpected", "inesperado",
        "surprise", "sorpresa", "taboo", "scandal", "escándalo", "hidden", "oculto", "exposed"
    ],
    "nostalgic": [
        "memory", "memoria", "remember", "recordar", "childhood", "infancia",
        "home", "hogar", "past", "pasado", "heritage", "tradición", "legacy", "legado"
    ]
}


# ============================================================
# PESOS DE JURY FIT POR CATEGORÍA
# ============================================================

CATEGORY_PROFILES = {
    "film": {
        "emotional": 0.25,
        "culture": 0.15,
        "viral": 0.10,
        "simplicity": 0.20,
        "execution": 0.20,
        "tech": 0.03,
        "social": 0.07
    },
    "film craft": {
        "emotional": 0.15,
        "culture": 0.10,
        "viral": 0.05,
        "simplicity": 0.10,
        "execution": 0.45,
        "tech": 0.05,
        "social": 0.10
    },
    "public relations": {
        "emotional": 0.08,
        "culture": 0.22,
        "viral": 0.35,
        "simplicity": 0.10,
        "execution": 0.08,
        "tech": 0.04,
        "social": 0.13
    },
    "pr": {
        "emotional": 0.08,
        "culture": 0.22,
        "viral": 0.35,
        "simplicity": 0.10,
        "execution": 0.08,
        "tech": 0.04,
        "social": 0.13
    },
    "outdoor": {
        "emotional": 0.08,
        "culture": 0.18,
        "viral": 0.28,
        "simplicity": 0.22,
        "execution": 0.14,
        "tech": 0.03,
        "social": 0.07
    },
    "innovation": {
        "emotional": 0.05,
        "culture": 0.12,
        "viral": 0.08,
        "simplicity": 0.10,
        "execution": 0.18,
        "tech": 0.32,
        "social": 0.15
    },
    "titanium": {
        "emotional": 0.15,
        "culture": 0.25,
        "viral": 0.20,
        "simplicity": 0.10,
        "execution": 0.10,
        "tech": 0.08,
        "social": 0.12
    },
    "brand experience": {
        "emotional": 0.10,
        "culture": 0.15,
        "viral": 0.22,
        "simplicity": 0.10,
        "execution": 0.25,
        "tech": 0.08,
        "social": 0.10
    },
    "direct": {
        "emotional": 0.08,
        "culture": 0.14,
        "viral": 0.15,
        "simplicity": 0.20,
        "execution": 0.10,
        "tech": 0.08,
        "social": 0.25
    },
    "design": {
        "emotional": 0.08,
        "culture": 0.12,
        "viral": 0.08,
        "simplicity": 0.20,
        "execution": 0.32,
        "tech": 0.08,
        "social": 0.12
    },
    "media": {
        "emotional": 0.08,
        "culture": 0.18,
        "viral": 0.30,
        "simplicity": 0.12,
        "execution": 0.10,
        "tech": 0.10,
        "social": 0.12
    },
    "default": {
        "emotional": 0.14,
        "culture": 0.18,
        "viral": 0.18,
        "simplicity": 0.15,
        "execution": 0.15,
        "tech": 0.08,
        "social": 0.12
    }
}


# ============================================================
# AUTOESTIMACIÓN SEMÁNTICA + REGLAS
# ============================================================

def semantic_best_label(text, prototypes, fallback_label):
    if embedding_model is None:
        return fallback_label, 0.0

    try:
        from sentence_transformers import util

        labels = list(prototypes.keys())
        prototype_texts = [prototypes[label] for label in labels]

        embeddings = embedding_model.encode(
            [text] + prototype_texts,
            convert_to_tensor=True
        )

        query_embedding = embeddings[0]
        prototype_embeddings = embeddings[1:]

        similarities = util.cos_sim(query_embedding, prototype_embeddings)[0].cpu().numpy()

        best_idx = int(np.argmax(similarities))

        return labels[best_idx], float(similarities[best_idx])

    except Exception:
        return fallback_label, 0.0


def keyword_idea_type(text, category):
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

    if count_keywords(full_text, IDEA_KEYWORDS["technology-driven idea"]) == 0:
        scores["technology-driven idea"] = 0

    priority = [
        "activism",
        "social impact",
        "public relations stunt",
        "experiential idea",
        "utility idea",
        "product innovation",
        "documentary",
        "humor",
        "absurdity",
        "emotional storytelling",
        "technology-driven idea",
        "innovation"
    ]

    max_score = max(scores.values())

    if max_score == 0:
        return "emotional storytelling"

    candidates = [k for k, v in scores.items() if v == max_score]

    for p in priority:
        if p in candidates:
            return p

    return candidates[0]


def keyword_sentiment(text, idea_type):
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

    priority = [
        "shocking",
        "funny",
        "tense",
        "hopeful",
        "nostalgic",
        "sad",
        "inspirational"
    ]

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


def get_category_profile(category):
    c = str(category).lower()

    for key, profile in CATEGORY_PROFILES.items():
        if key != "default" and key in c:
            return profile

    return CATEGORY_PROFILES["default"]


def estimate_scores(project_name, description, category, idea_type, sentiment):
    text = f"{project_name} {description} {category} {idea_type} {sentiment}".lower()

    social_words = IDEA_KEYWORDS["social impact"] + IDEA_KEYWORDS["activism"]

    viral_words = IDEA_KEYWORDS["public relations stunt"] + [
        "viral",
        "share",
        "conversation",
        "conversación",
        "famoso",
        "noticia",
        "medios",
        "prensa"
    ]

    tech_words = IDEA_KEYWORDS["technology-driven idea"]

    complexity_words = [
        "installation",
        "instalación",
        "event",
        "evento",
        "live",
        "en vivo",
        "platform",
        "plataforma",
        "app",
        "technology",
        "tecnología",
        "film",
        "documentary",
        "documental",
        "production",
        "producción",
        "metro",
        "aeropuerto",
        "aeropuertos",
        "plazas",
        "ciudad",
        "nacional"
    ]

    simplicity_words = [
        "simple",
        "simplicity",
        "clarity",
        "claro",
        "directo",
        "fácil",
        "easy",
        "one idea",
        "memorable",
        "ola",
        "unidos",
        "unir"
    ]

    social_count = count_keywords(text, social_words)
    viral_count = count_keywords(text, viral_words)
    tech_count = count_keywords(text, tech_words)
    complexity_count = count_keywords(text, complexity_words)
    simplicity_count = count_keywords(text, simplicity_words)

    cultural_relevance = 4.5 + min(4.5, social_count * 1.0)
    viral_potential = 4.5 + min(4.5, viral_count * 1.0)
    simplicity_of_insight = 6.0 + min(3.0, simplicity_count * 0.8)
    execution_complexity = 4.0 + min(5.0, complexity_count * 0.7)

    tech_component_ai = 1 if tech_count > 0 or idea_type == "technology-driven idea" else 0
    social_impact_ai = 1 if social_count > 0 or idea_type in ["social impact", "activism"] else 0

    if idea_type == "public relations stunt":
        viral_potential += 1.5
        execution_complexity += 0.8

    if idea_type == "experiential idea":
        viral_potential += 0.8
        execution_complexity += 1.2

    if idea_type == "emotional storytelling":
        cultural_relevance += 0.8
        simplicity_of_insight += 0.6

    if idea_type == "technology-driven idea":
        execution_complexity += 2

    if idea_type == "utility idea":
        simplicity_of_insight += 0.5

    if idea_type == "product innovation":
        execution_complexity += 1

    if idea_type in ["social impact", "activism"]:
        cultural_relevance += 1.5

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

    if "media" in c:
        viral_potential += 0.5

    return {
        "cultural_relevance": round(clamp(cultural_relevance, 1, 10), 1),
        "viral_potential": round(clamp(viral_potential, 1, 10), 1),
        "simplicity_of_insight": round(clamp(simplicity_of_insight, 1, 10), 1),
        "execution_complexity": round(clamp(execution_complexity, 1, 10), 1),
        "tech_component_ai": int(tech_component_ai),
        "social_impact_ai": int(social_impact_ai)
    }


def estimate_jury_fit(category, idea_type, sentiment, scores):
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
    if not has_enough_creative_information(project_name, description):
        return {
            "idea_type_ai": "emotional storytelling",
            "sentiment_ai": "hopeful",
            "cultural_relevance": 2.0,
            "viral_potential": 2.0,
            "simplicity_of_insight": 3.0,
            "execution_complexity": 2.0,
            "tech_component_ai": 0,
            "social_impact_ai": 0,
            "jury_fit_score": 25.0,
            "jury_fit_status": "baja afinidad",
            "classification_confidence": 0.20,
            "semantic_idea_confidence": 0.0,
            "semantic_sentiment_confidence": 0.0
        }

    text = f"{project_name}. {description}. Category: {category}"

    keyword_idea = keyword_idea_type(text, category)
    semantic_idea, semantic_idea_conf = semantic_best_label(text, IDEA_PROTOTYPES, keyword_idea)

    if semantic_idea_conf >= 0.25:
        idea_type = semantic_idea
    else:
        idea_type = keyword_idea

    keyword_sent = keyword_sentiment(text, idea_type)
    semantic_sent, semantic_sent_conf = semantic_best_label(text, SENTIMENT_PROTOTYPES, keyword_sent)

    if semantic_sent_conf >= 0.20:
        sentiment = semantic_sent
    else:
        sentiment = keyword_sent

    scores = estimate_scores(project_name, description, category, idea_type, sentiment)
    jury_fit_score = estimate_jury_fit(category, idea_type, sentiment, scores)
    jury_fit_status = fit_status_from_score(jury_fit_score)

    confidence = 0.55

    if semantic_idea_conf >= 0.35:
        confidence += 0.10

    if semantic_sent_conf >= 0.30:
        confidence += 0.10

    if len(description.split()) >= 35:
        confidence += 0.10

    confidence = round(clamp(confidence, 0.20, 0.90), 2)

    return {
        "idea_type_ai": idea_type,
        "sentiment_ai": sentiment,
        **scores,
        "jury_fit_score": jury_fit_score,
        "jury_fit_status": jury_fit_status,
        "classification_confidence": confidence,
        "semantic_idea_confidence": round(float(semantic_idea_conf), 3),
        "semantic_sentiment_confidence": round(float(semantic_sent_conf), 3)
    }




# ============================================================
# SCORE DIRECTO BASADO EN TEXTO + PENALTY ENGINE CANNES
# ============================================================

# Señales de publicidad commodity.
# Importante: NO incluimos palabras como medios, pantallas, espectaculares, campaña,
# publicidad o llamado a acción, porque en Cannes pueden ser parte de una activación válida.
WEAK_COMMODITY_SIGNALS = [
    "post en instagram", "instagram post", "posteo", "social media post",
    "social media posts", "posts de redes", "banner", "display ads",
    "anuncio display", "flyer", "volante", "email blast", "newsletter",
    "landing page", "logo grande", "foto del producto", "product shot",
    "comercial tradicional", "spot tradicional", "catálogo", "catalogo",
    "performance", "ecommerce", "retail promotion"
]

HARD_COMMODITY_SIGNALS = [
    "discount", "descuento", "sale", "oferta", "promo", "promoción", "promocion",
    "20%", "30%", "40%", "50%", "2x1", "limited time", "tiempo limitado",
    "solo por hoy", "buy now", "compra ahora", "cupón", "cupon", "rebaja",
    "black friday", "hot sale", "giveaway", "sorteo"
]

CANNES_PROTECTIVE_SIGNALS = [
    "récord", "record", "guinness", "historia", "participantes", "personas reales",
    "participación", "participacion", "co-creación", "co-creacion", "crowdsourced",
    "metro", "aeropuerto", "aeropuertos", "ciudad", "ciudades", "espacio público",
    "espacio publico", "copa del mundo", "mundial", "méxico", "mexico", "país", "pais",
    "cultural", "tensión", "tension", "decepción", "decepcion", "orgullo",
    "noticia", "prensa", "earned media", "conversación", "conversacion", "viral",
    "en tiempo real", "contador", "activación", "activacion", "instalación", "instalacion"
]

CANNES_POSITIVE_SIGNALS = {
    "cultural_tension": [
        "tensión cultural", "tension cultural", "cultural tension", "problema cultural",
        "tabú", "taboo", "prejuicio", "estigma", "debate", "conversación social",
        "conversacion social", "movimiento", "movement", "causa", "inequality", "desigualdad",
        "discriminación", "discriminacion", "derechos", "rights", "climate", "clima",
        "decepción", "decepcion", "orgullo", "país", "pais", "méxico", "mexico"
    ],
    "earned_media": [
        "earned media", "prensa", "medios", "noticia", "news", "viral", "share",
        "compartir", "conversación", "conversacion", "buzz", "public conversation",
        "talk of", "trend", "trending", "récord", "record", "world's first", "first ever",
        "primera vez", "nunca antes", "más grande", "mas grande", "historia"
    ],
    "participation": [
        "participa", "participación", "participacion", "invita a la gente", "usuarios",
        "comunidad", "community", "crowdsourced", "co-creación", "co-creacion",
        "interactive", "interactivo", "reto", "challenge", "activación", "activacion",
        "instalación", "instalacion", "experiencia", "experience", "graba", "publica",
        "personas reales", "cualquier persona"
    ],
    "craft_or_execution": [
        "film", "documental", "documentary", "instalación", "instalacion", "prototype",
        "prototipo", "plataforma", "app", "ai", "inteligencia artificial", "data",
        "sensor", "realidad aumentada", "augmented reality", "diseño", "design",
        "packaging", "producto", "product innovation", "pantallas", "contador",
        "en tiempo real", "compilamos", "editamos", "aeropuertos", "metro"
    ],
    "human_truth": [
        "historia real", "real story", "testimonio", "testimony", "familia", "family",
        "memoria", "memory", "identidad", "identity", "orgullo", "pride", "sueño", "dream",
        "vida", "life", "personas", "people", "human", "humano", "decepción", "decepcion",
        "selección", "seleccion", "mexicanos"
    ]
}

WEAK_OR_GENERIC_SIGNALS = [
    "innovador", "innovadora", "disruptivo", "disruptiva", "impactante", "creativo",
    "creativa", "único", "unico", "emocionante", "diferente", "gran campaña",
    "gran campana", "experiencia increíble", "experiencia increible", "muy original",
    "llamativo", "atractivo", "moderno", "premium", "aspiracional"
]


def keyword_hits(text, words):
    text = str(text).lower()
    hits = []
    for w in words:
        if str(w).lower() in text:
            hits.append(w)
    return hits


def get_dimension_raw_hits(project_name, description):
    text = f"{project_name} {description}".lower()
    return {dimension: keyword_hits(text, terms) for dimension, terms in CANNES_POSITIVE_SIGNALS.items()}


def calculate_commodity_penalty(project_name, description, originality_score=None, dimension_scores=None):
    """
    Penaliza publicidad commodity sin castigar activaciones con medios.
    Una pantalla, un espectacular o un CTA no son commodity por sí mismos;
    commodity es que la idea dependa de descuento, post, banner, logo o promoción retail.
    """
    text = f"{project_name} {description}".lower()
    words = [w for w in re.findall(r"\b\w+\b", text) if len(w) > 2]

    weak_hits = keyword_hits(text, WEAK_COMMODITY_SIGNALS)
    hard_hits = keyword_hits(text, HARD_COMMODITY_SIGNALS)
    generic_hits = keyword_hits(text, WEAK_OR_GENERIC_SIGNALS)
    protective_hits = keyword_hits(text, CANNES_PROTECTIVE_SIGNALS)

    penalty = 0
    reasons = []

    if len(words) < 18:
        penalty += 22
        reasons.append("La descripción es demasiado corta: no permite ver una idea Cannes completa.")
    elif len(words) < 35:
        penalty += 12
        reasons.append("La descripción es breve; falta tensión, mecánica, ejecución e impacto.")

    raw_commodity = len(weak_hits) * 4 + len(hard_hits) * 10

    promo_core = any(x in text for x in [
        "descuento", "discount", "promo", "promoción", "promocion", "oferta",
        "sale", "2x1", "rebaja", "cupón", "cupon"
    ])
    ad_format = any(x in text for x in [
        "post en instagram", "instagram post", "banner", "flyer", "anuncio display",
        "logo grande", "foto del producto", "product shot", "compra ahora", "buy now"
    ])

    if promo_core and ad_format:
        raw_commodity += 25
        reasons.append("La mecánica parece depender de promoción + formato publicitario básico.")

    if len(generic_hits) >= 3 and len(words) < 70:
        raw_commodity += 8
        reasons.append("Usa lenguaje genérico sin explicar suficientemente la mecánica creativa.")

    cannes_protection = 0
    if originality_score is not None and originality_score >= 65:
        cannes_protection += 1
    if dimension_scores:
        if dimension_scores.get("earned_media", 0) >= 12:
            cannes_protection += 1
        if dimension_scores.get("participation", 0) >= 12:
            cannes_protection += 1
        if dimension_scores.get("cultural_tension", 0) >= 7:
            cannes_protection += 1
    if len(protective_hits) >= 3:
        cannes_protection += 1

    # Si hay señales Cannes fuertes, reducimos el castigo por palabras publicitarias sueltas.
    if cannes_protection >= 3:
        raw_commodity *= 0.25
    elif cannes_protection == 2:
        raw_commodity *= 0.50

    penalty += raw_commodity

    if weak_hits or hard_hits:
        detected = hard_hits + weak_hits
        reasons.append("Señales commodity detectadas: " + ", ".join(detected[:6]) + ".")

    if cannes_protection >= 3 and (weak_hits or hard_hits):
        reasons.append("La penalización commodity fue reducida porque el texto sí contiene participación, earned media o tensión cultural.")

    return round(clamp(penalty, 0, 70), 1), reasons, hard_hits + weak_hits

def calculate_cannes_originality_score(project_name, description):
    """
    Evalúa si el texto contiene señales reales de una idea tipo Cannes:
    tensión cultural, earned media, participación, craft/ejecución y verdad humana.
    """
    text = f"{project_name} {description}".lower()
    words = [w for w in re.findall(r"\b\w+\b", text) if len(w) > 2]

    score = 0
    reasons = []
    dimension_scores = {}

    for dimension, terms in CANNES_POSITIVE_SIGNALS.items():
        hits = keyword_hits(text, terms)
        if len(hits) >= 3:
            points = 16
        elif len(hits) == 2:
            points = 12
        elif len(hits) == 1:
            points = 7
        else:
            points = 0
        dimension_scores[dimension] = points
        score += points

    if len(words) >= 90:
        score += 14
        reasons.append("La descripción tiene profundidad suficiente para evaluar la mecánica.")
    elif len(words) >= 55:
        score += 10
        reasons.append("La descripción tiene detalle razonable.")
    elif len(words) >= 35:
        score += 6
        reasons.append("La descripción se entiende, pero todavía necesita más profundidad.")
    else:
        score += 2
        reasons.append("La descripción no desarrolla suficiente la idea.")

    if dimension_scores["cultural_tension"] == 0:
        reasons.append("No se detecta tensión cultural clara.")
    else:
        reasons.append("Hay señal de tensión/relevancia cultural.")

    if dimension_scores["earned_media"] == 0:
        reasons.append("No se detecta una razón fuerte para generar earned media o conversación.")
    else:
        reasons.append("Hay potencial de conversación o earned media.")

    if dimension_scores["participation"] == 0:
        reasons.append("No se detecta participación pública, activación o experiencia clara.")
    else:
        reasons.append("Hay una mecánica participativa/experiencial.")

    if dimension_scores["human_truth"] > 0:
        reasons.append("Hay una verdad humana o emocional identificable.")

    return round(clamp(score, 0, 100), 1), reasons, dimension_scores


def calculate_text_quality_score(project_name, description, category):
    """
    Score directo del texto. Combina variables autoestimadas con un penalty engine.
    La idea: el modelo puede opinar, pero el texto pone límites duros.
    """
    text = f"{project_name} {description}".lower().strip()
    words = [w for w in re.findall(r"\b\w+\b", text) if len(w) > 2]

    if len(words) < 8:
        return 12.0, ["Descripción demasiado corta: no hay suficiente idea para evaluar."], 70, 0.0

    auto = auto_estimate_all(project_name, description, category)
    originality_score, originality_reasons, dimension_scores = calculate_cannes_originality_score(project_name, description)
    commodity_penalty, penalty_reasons, bad_hits = calculate_commodity_penalty(
        project_name,
        description,
        originality_score=originality_score,
        dimension_scores=dimension_scores
    )

    # Base textual más conservadora que antes.
    score = 0
    reasons = []

    # 1) Originalidad Cannes pesa mucho más que categoría.
    score += originality_score * 0.48

    # 2) Variables inferidas pesan, pero con menor peso para no sobrepremiar keywords.
    score += auto["cultural_relevance"] * 1.4
    score += auto["viral_potential"] * 1.4
    score += auto["simplicity_of_insight"] * 1.1

    # 3) Ejecución suma solo si hay algo que ejecutar.
    if auto["execution_complexity"] >= 5 and originality_score >= 35:
        score += min(auto["execution_complexity"] * 1.0, 8)
    else:
        score += 2

    # 4) Bonus Cannes moderados.
    if auto["social_impact_ai"] == 1 and originality_score >= 35:
        score += 6
        reasons.append("Detecta impacto social con algo de sustancia creativa.")

    if auto["tech_component_ai"] == 1 and originality_score >= 35:
        score += 5
        reasons.append("Detecta componente tecnológico con potencial creativo.")

    if auto["jury_fit_score"] >= 75 and originality_score >= 45:
        score += 6
        reasons.append("Buena afinidad con categoría/jurado, respaldada por señales del texto.")
    elif auto["jury_fit_score"] >= 55:
        score += 3
        reasons.append("Afinidad media con la categoría, pero no suficiente por sí sola.")

    # 5) Penalizaciones fuertes.
    score -= commodity_penalty

    reasons.extend(originality_reasons)
    reasons.extend(penalty_reasons)

    # Caps duros: si la idea es commodity, no puede llegar a shortlist.
    if commodity_penalty >= 55:
        score = min(score, 38)
        reasons.append("Cap aplicado: una idea commodity fuerte no puede superar rango bajo sin una mecánica cultural excepcional.")
    elif commodity_penalty >= 35:
        score = min(score, 48)
        reasons.append("Cap aplicado: demasiadas señales promocionales para rango shortlist.")
    elif originality_score < 25:
        score = min(score, 45)
        reasons.append("Cap aplicado: baja originalidad Cannes detectada en el texto.")
    elif originality_score < 40 and commodity_penalty >= 20:
        score = min(score, 55)
        reasons.append("Cap aplicado: originalidad limitada + señales promocionales.")

    final_score = round(clamp(score, 0, 100), 1)
    return final_score, reasons, commodity_penalty, originality_score


def blend_model_with_text_score(model_score, model_probability, text_score, description, commodity_penalty=0, originality_score=0):
    """
    Mezcla la predicción del modelo con el score textual.
    El texto ahora tiene más control que el RF cuando detecta baja originalidad o publicidad commodity.
    """
    words = [w for w in re.findall(r"\b\w+\b", str(description).lower()) if len(w) > 2]

    if commodity_penalty >= 55 or originality_score < 25:
        text_weight = 0.75
        max_score = 48
        max_probability = 0.22
    elif commodity_penalty >= 35:
        text_weight = 0.65
        max_score = 58
        max_probability = 0.32
    elif len(words) < 35:
        text_weight = 0.55
        max_score = 65
        max_probability = 0.42
    elif originality_score < 40:
        text_weight = 0.55
        max_score = 68
        max_probability = 0.50
    else:
        text_weight = 0.40
        max_score = 100
        max_probability = 0.90

    blended_score = (1 - text_weight) * model_score + text_weight * text_score
    blended_probability = (1 - text_weight) * model_probability + text_weight * (text_score / 100)

    blended_score = min(blended_score, max_score)
    blended_probability = min(blended_probability, max_probability)

    return round(float(clamp(blended_score, 0, 100)), 1), round(float(clamp(blended_probability, 0, 1)), 3)


def strict_score_label(score):
    """Etiqueta más estricta para que shortlist no empiece demasiado bajo."""
    if score >= 92:
        return "Potencial Gold / Grand Prix"
    elif score >= 84:
        return "Potencial metal"
    elif score >= 76:
        return "Potencial shortlist / bronze"
    elif score >= 60:
        return "Potencial medio, necesita fortalecer idea"
    return "Potencial bajo"

# ============================================================
# PREDICCIÓN
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
# RECOMENDACIONES ESTRATÉGICAS BASADAS EN TEXTO
# ============================================================

def generate_cannes_recommendations(project_name, description, category, input_row, predicted_score, prob_high_award, originality_score, commodity_penalty):
    """
    Genera recomendaciones accionables para mejorar la idea con lógica de Cannes.
    No usa IA generativa externa; usa reglas interpretables basadas en texto, categoría y scores.
    """
    text = f"{project_name} {description}".lower()
    category_l = str(category).lower()
    dim_hits = get_dimension_raw_hits(project_name, description)

    recommendations = []
    strengths = []

    # Fortalezas detectadas
    if len(dim_hits.get("cultural_tension", [])) > 0:
        strengths.append("Tiene una tensión cultural o contexto país que puede sostener una narrativa Cannes.")
    if len(dim_hits.get("earned_media", [])) > 0:
        strengths.append("Tiene potencial de conversación pública / earned media.")
    if len(dim_hits.get("participation", [])) > 0:
        strengths.append("La mecánica invita a participación, no solo a exposición publicitaria.")
    if len(dim_hits.get("craft_or_execution", [])) > 0:
        strengths.append("Hay una base de ejecución visible: medios, plataforma, instalación, data o experiencia.")
    if len(dim_hits.get("human_truth", [])) > 0:
        strengths.append("Tiene una verdad humana/emocional que puede convertir la idea en historia.")

    # Recomendaciones universales
    if not any(w in text for w in ["guinness", "récord oficial", "record oficial", "validación", "validacion", "certificación", "certificacion"]):
        recommendations.append({
            "title": "Convertir la idea en un hecho verificable",
            "why": "La idea habla de lograr algo histórico, pero necesita una prueba clara para que prensa, jurado y público lo crean.",
            "how": [
                "Definir si será récord oficial, récord de marca o récord cultural documentado.",
                "Agregar validación externa: Guinness World Records, notario, plataforma pública de conteo o alianza con medios.",
                "Mostrar contador en tiempo real, número de participantes, ciudades y kilómetros simbólicos recorridos."
            ]
        })

    if not any(w in text for w in ["marca", "brand", "patrocinador", "sponsor", "habilita", "presenta", "powered by"]):
        recommendations.append({
            "title": "Hacer inevitable el rol de la marca",
            "why": "Una idea Cannes fuerte no solo es buena culturalmente; la marca debe tener derecho a firmarla.",
            "how": [
                "Explicar por qué esta marca puede unir a la gente mejor que nadie.",
                "Dar a la marca un rol funcional: plataforma, tecnología, infraestructura, acceso, data o convocatoria.",
                "Evitar que parezca una idea que cualquier institución podría firmar igual."
            ]
        })

    if input_row.get("viral_potential", 0) < 8 or len(dim_hits.get("earned_media", [])) < 2:
        recommendations.append({
            "title": "Diseñar el momento noticioso",
            "why": "La idea puede ser participativa, pero Cannes premia cuando la campaña se vuelve noticia por una razón concreta.",
            "how": [
                "Definir el día exacto del rompimiento o lanzamiento como evento nacional.",
                "Crear un titular fácil: ‘México hace la ola más grande del mundo aunque no llegue a la final’." ,
                "Preparar assets para prensa: visualizador en vivo, mapa, ranking por ciudad y video hero de 60 segundos."
            ]
        })

    if input_row.get("execution_complexity", 0) >= 8:
        recommendations.append({
            "title": "Bajar el riesgo de ejecución",
            "why": "La idea es grande; si la mecánica no está clara, el jurado puede verla como aspiracional pero difícil de probar.",
            "how": [
                "Separar la ejecución en fases: teaser, convocatoria, acumulación, rompimiento y film final.",
                "Definir reglas simples para participar: duración, formato, hashtag, geolocalización y moderación.",
                "Agregar un prototipo visual de cómo se une cada video en una sola ola continua."
            ]
        })

    if originality_score < 75:
        recommendations.append({
            "title": "Agregar un twist propio que nadie más pueda copiar",
            "why": "La base es buena, pero necesita un giro diferencial para sentirse más Cannes y menos activación masiva estándar.",
            "how": [
                "Que la ola reaccione en tiempo real a participación por ciudad, país o selección eliminada.",
                "Convertir la decepción deportiva en una acción emocional: ‘el equipo no llegó, México sí’." ,
                "Hacer que cada pantalla pública sea un fragmento vivo de la ola, no solo un medio que la comunica."
            ]
        })

    if commodity_penalty > 8:
        recommendations.append({
            "title": "Reducir cualquier lectura de publicidad tradicional",
            "why": "El sistema detectó algunas señales commodity. No necesariamente dañan la idea, pero hay que evitar que parezca solo pauta o convocatoria.",
            "how": [
                "Cambiar lenguaje de ‘soportes publicitarios’ por ‘infraestructura viva de participación’." ,
                "Mostrar cómo el medio se transforma con la ola, no solo cómo muestra anuncios.",
                "Evitar que el call to action sea el centro; el centro debe ser el acontecimiento cultural."
            ]
        })

    # Recomendaciones por categoría
    if "public relations" in category_l or category_l.strip() == "pr":
        recommendations.append({
            "title": "Para PR: construir la estrategia de earned media",
            "why": "En PR no basta una activación; debe existir una noticia que viaje sola.",
            "how": [
                "Definir 3 titulares de prensa antes de producir la campaña.",
                "Incluir voceros, datos en vivo y alianzas con medios deportivos/culturales.",
                "Preparar una escalera de conversación: México, fans, diáspora, Mundial, récord."
            ]
        })
    elif "outdoor" in category_l:
        recommendations.append({
            "title": "Para Outdoor: hacer que el medio sea la idea",
            "why": "En Outdoor, el soporte debe transformar el espacio físico, no solo alojar contenido.",
            "how": [
                "Sincronizar pantallas para que la ola avance físicamente de izquierda a derecha.",
                "Usar estaciones, túneles, aeropuertos o puentes como tramos reales de la ola.",
                "Medir participación por ubicación y reflejarla en los medios en vivo."
            ]
        })
    elif "titanium" in category_l:
        recommendations.append({
            "title": "Para Titanium: elevarlo de campaña a movimiento cultural",
            "why": "Titanium premia ideas que cambian el comportamiento o la conversación de una categoría.",
            "how": [
                "Convertir la ola en un símbolo nacional que trascienda a un partido.",
                "Crear participación internacional de mexicanos en el extranjero.",
                "Demostrar impacto cultural: participación, prensa, conversación, uso orgánico y permanencia."
            ]
        })
    elif "innovation" in category_l:
        recommendations.append({
            "title": "Para Innovation: fortalecer la tecnología detrás de la idea",
            "why": "Innovation necesita defensibilidad técnica, no solo una ejecución digital bonita.",
            "how": [
                "Explicar el motor que une videos: moderación, stitching, geolocalización y render en tiempo real.",
                "Crear una demo del sistema antes del caso final.",
                "Mostrar qué parte de la tecnología es nueva o difícil de replicar."
            ]
        })
    elif "film" in category_l:
        recommendations.append({
            "title": "Para Film: diseñar el arco emocional",
            "why": "Film necesita una historia que se sienta inevitable, no solo una compilación de participación.",
            "how": [
                "Abrir con la tensión: México quizá no llega a la final, pero la gente sí.",
                "Seguir con rostros reales: familias, oficinas, metro, aeropuertos, azoteas.",
                "Cerrar con la ola unificada y una frase memorable."
            ]
        })

    # Limitar duplicados por título
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec["title"] not in seen:
            unique_recommendations.append(rec)
            seen.add(rec["title"])

    return strengths, unique_recommendations[:7]


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================

st.title("🦁 Cannes Creative Intelligence")

st.caption(
    "Modelo exploratorio para estimar potencial Cannes a partir de descripción, variables creativas, "
    "categoría, jurado y afinidad. No usa marca ni agencia como predictor."
)

with st.sidebar:
    st.header("Configuración")
    st.info("Escribe una descripción clara, usa Autoestimar con IA abierta y ajusta manualmente si hace falta.")
    show_importance = st.checkbox("Mostrar importancia de variables", value=True)
    show_jury_details = st.checkbox("Mostrar jurado y afinidades", value=True)
    st.caption("Autoestimación semántica: sentence-transformers/all-MiniLM-L6-v2")

category_options = training_categories.get("category_norm", []) or [
    "Film",
    "Outdoor",
    "Titanium",
    "Innovation",
    "Public Relations",
    "Brand Experience & Activation",
    "Media",
    "Design",
    "Direct"
]

subcategory_options = training_categories.get("subcategory", []) or ["Unknown"]
idea_type_options = training_categories.get("idea_type_ai", []) or list(IDEA_PROTOTYPES.keys())
sentiment_options = training_categories.get("sentiment_ai", []) or list(SENTIMENT_PROTOTYPES.keys())

DEFAULTS = {
    "idea_type_ai_select": "emotional storytelling",
    "sentiment_ai_select": "hopeful",
    "cultural_relevance": 3.0,
    "viral_potential": 2.0,
    "simplicity_of_insight": 4.0,
    "execution_complexity": 2.0,
    "tech_component_ai": 0,
    "social_impact_ai": 0,
    "jury_fit_score": 30.0,
    "classification_confidence": 0.35,
    "semantic_idea_confidence": 0.0,
    "semantic_sentiment_confidence": 0.0
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
        "Describe qué sucede, dónde vive la campaña, qué problema o tensión cultural aborda, "
        "por qué sería relevante y cómo generaría conversación.",
        height=160
    )

    category_norm = st.selectbox("Categoría Cannes", category_options)
    subcategory = st.selectbox("Subcategoría", subcategory_options)

    jury_profile = get_jury_profile(category_norm)

    if show_jury_details:
        st.markdown("### Jurado / afinidad de categoría")

        if jury_profile is not None:
            st.write(f"**Jurado:** {jury_profile.get('jury_president_name', 'No disponible')}")
            st.write(f"**Rol:** {jury_profile.get('jury_role', 'No disponible')}")
            st.write(f"**Afinidad general:** {jury_profile.get('affinity_summary', 'No disponible')}")

            affinities = [
                jury_profile.get("affinity_1", ""),
                jury_profile.get("affinity_2", ""),
                jury_profile.get("affinity_3", ""),
                jury_profile.get("affinity_4", "")
            ]

            affinities = [a for a in affinities if str(a).strip() != ""]

            if affinities:
                st.write("**Qué suele valorar:**")
                for affinity in affinities:
                    st.write(f"• {affinity}")

            watchout = jury_profile.get("watchout", "")

            if str(watchout).strip() != "":
                st.warning(f"Watchout: {watchout}")

        else:
            st.info("No hay perfil específico de jurado para esta categoría. Se usará el perfil general de afinidad por categoría.")

    if not has_enough_creative_information(project_name, project_description):
        st.warning(
            "La descripción todavía es muy corta o genérica. Agrega qué sucede, dónde vive la idea, "
            "qué la hace relevante y por qué generaría conversación."
        )

    if st.button("Autoestimar variables con IA abierta", type="secondary"):
        auto = auto_estimate_all(project_name, project_description, category_norm)

        st.session_state["idea_type_ai_select"] = auto["idea_type_ai"]
        st.session_state["sentiment_ai_select"] = auto["sentiment_ai"]
        st.session_state["cultural_relevance"] = auto["cultural_relevance"]
        st.session_state["viral_potential"] = auto["viral_potential"]
        st.session_state["simplicity_of_insight"] = auto["simplicity_of_insight"]
        st.session_state["execution_complexity"] = auto["execution_complexity"]
        st.session_state["tech_component_ai"] = auto["tech_component_ai"]
        st.session_state["social_impact_ai"] = auto["social_impact_ai"]
        st.session_state["jury_fit_score"] = auto["jury_fit_score"]
        st.session_state["classification_confidence"] = auto["classification_confidence"]
        st.session_state["semantic_idea_confidence"] = auto["semantic_idea_confidence"]
        st.session_state["semantic_sentiment_confidence"] = auto["semantic_sentiment_confidence"]

        st.rerun()

with col2:
    if st.session_state["idea_type_ai_select"] not in idea_type_options:
        idea_type_options = [st.session_state["idea_type_ai_select"]] + idea_type_options

    if st.session_state["sentiment_ai_select"] not in sentiment_options:
        sentiment_options = [st.session_state["sentiment_ai_select"]] + sentiment_options

    idea_type_ai = st.selectbox(
        "Tipo de idea",
        idea_type_options,
        key="idea_type_ai_select"
    )

    sentiment_ai = st.selectbox(
        "Sentimiento principal",
        sentiment_options,
        key="sentiment_ai_select"
    )

    st.caption(
        f"Confianza semántica idea: {st.session_state.get('semantic_idea_confidence', 0):.3f} · "
        f"Confianza semántica sentimiento: {st.session_state.get('semantic_sentiment_confidence', 0):.3f}"
    )

st.divider()
st.markdown("### Variables creativas autoestimadas / editables")

col3, col4 = st.columns(2)

with col3:
    cultural_relevance = st.slider(
        "Relevancia cultural",
        1.0,
        10.0,
        float(st.session_state["cultural_relevance"]),
        0.5
    )

    viral_potential = st.slider(
        "Potencial viral / PR",
        1.0,
        10.0,
        float(st.session_state["viral_potential"]),
        0.5
    )

    simplicity_of_insight = st.slider(
        "Simplicidad del insight",
        1.0,
        10.0,
        float(st.session_state["simplicity_of_insight"]),
        0.5
    )

    execution_complexity = st.slider(
        "Complejidad de ejecución",
        1.0,
        10.0,
        float(st.session_state["execution_complexity"]),
        0.5
    )

with col4:
    tech_component_ai = st.radio(
        "Componente tecnológico",
        [0, 1],
        index=int(st.session_state["tech_component_ai"]),
        format_func=lambda x: "Sí" if x == 1 else "No",
        horizontal=True
    )

    social_impact_ai = st.radio(
        "Impacto social",
        [0, 1],
        index=int(st.session_state["social_impact_ai"]),
        format_func=lambda x: "Sí" if x == 1 else "No",
        horizontal=True
    )

    current_scores = {
        "cultural_relevance": cultural_relevance,
        "viral_potential": viral_potential,
        "simplicity_of_insight": simplicity_of_insight,
        "execution_complexity": execution_complexity,
        "tech_component_ai": tech_component_ai,
        "social_impact_ai": social_impact_ai
    }

    if st.button("Recalcular Jury Fit con variables actuales"):
        st.session_state["jury_fit_score"] = estimate_jury_fit(
            category_norm,
            idea_type_ai,
            sentiment_ai,
            current_scores
        )
        st.rerun()

    jury_fit_score = st.slider(
        "Jury Fit Score",
        0.0,
        100.0,
        float(st.session_state["jury_fit_score"]),
        1.0
    )

    jury_fit_status = fit_status_from_score(jury_fit_score)
    st.write(f"Afinidad: **{jury_fit_status}**")

classification_confidence = st.slider(
    "Confianza de clasificación",
    0.0,
    1.0,
    float(st.session_state["classification_confidence"]),
    0.05
)

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
    base_score, base_prob_high_award = predict_campaign_potential(input_row)
    text_quality_score, text_reasons, commodity_penalty, originality_score = calculate_text_quality_score(
        project_name,
        project_description,
        category_norm
    )

    predicted_score, prob_high_award = blend_model_with_text_score(
        base_score,
        base_prob_high_award,
        text_quality_score,
        project_description,
        commodity_penalty,
        originality_score
    )

    label = strict_score_label(predicted_score)
    strengths, risks = explain_campaign(input_row, predicted_score, prob_high_award)

    st.divider()
    st.subheader(project_name)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Cannes Score estimado", f"{predicted_score}/100")
    metric2.metric("Probabilidad High Award", f"{prob_high_award * 100:.1f}%")
    metric3.metric("Etiqueta", label)

    st.caption(
        f"Score base del modelo: {base_score}/100 · "
        f"Probabilidad base: {base_prob_high_award * 100:.1f}% · "
        f"Score textual: {text_quality_score}/100 · Originalidad Cannes: {originality_score}/100 · Penalización commodity: {commodity_penalty}/70"
    )

    st.markdown("### Lectura estratégica")

    if text_reasons:
        st.markdown("**Lectura del texto**")
        for reason in text_reasons:
            st.write(f"• {reason}")

    if jury_profile is not None:
        st.markdown("**Contexto de jurado / categoría**")
        st.write(f"Jurado: {jury_profile.get('jury_president_name', 'No disponible')}")
        st.write(f"Afinidad esperada: {jury_profile.get('affinity_summary', 'No disponible')}")

    if strengths:
        st.markdown("**Fortalezas**")
        for item in strengths:
            st.write(f"✅ {item}")

    if risks:
        st.markdown("**Riesgos / áreas a mejorar**")
        for item in risks:
            st.write(f"⚠️ {item}")

    detected_strengths, recommendations = generate_cannes_recommendations(
        project_name,
        project_description,
        category_norm,
        input_row,
        predicted_score,
        prob_high_award,
        originality_score,
        commodity_penalty
    )

    st.markdown("### Recomendaciones para mejorar la idea")

    if detected_strengths:
        st.markdown("**Lo que ya está funcionando**")
        for item in detected_strengths:
            st.write(f"✅ {item}")

    if recommendations:
        st.markdown("**Qué reforzar para subir potencial Cannes**")
        for i, rec in enumerate(recommendations, start=1):
            with st.expander(f"{i}. {rec['title']}", expanded=i <= 3):
                st.write(f"**Por qué importa:** {rec['why']}")
                st.write("**Cómo mejorarlo:**")
                for action in rec["how"]:
                    st.write(f"• {action}")
    else:
        st.info("No se detectaron recomendaciones específicas. Agrega más detalle sobre mecánica, ejecución, rol de marca e impacto esperado.")

    recommendation_text = " | ".join([
        rec["title"] + ": " + "; ".join(rec["how"]) for rec in recommendations
    ])

    result_df = pd.DataFrame([{
        "project_name": project_name,
        "project_description": project_description,
        "jury_president_name": jury_profile.get("jury_president_name", "") if jury_profile else "",
        "jury_role": jury_profile.get("jury_role", "") if jury_profile else "",
        "jury_affinity_summary": jury_profile.get("affinity_summary", "") if jury_profile else "",
        **input_row,
        "base_model_score": base_score,
        "base_prob_high_award": base_prob_high_award,
        "text_quality_score": text_quality_score,
        "cannes_originality_score": originality_score,
        "commodity_advertising_penalty": commodity_penalty,
        "predicted_cannes_score": predicted_score,
        "prob_high_award": prob_high_award,
        "prob_high_award_percent": round(prob_high_award * 100, 1),
        "potential_label": label,
        "strengths": "; ".join(strengths),
        "risks": "; ".join(risks),
        "detected_text_strengths": "; ".join(detected_strengths),
        "recommendations": recommendation_text
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

        pred_df = pd.concat(
            [ideas_df.reset_index(drop=True), pd.DataFrame(predictions)],
            axis=1
        )

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

st.caption(
    "Modelo exploratorio desarrollado para ISA. No debe interpretarse como garantía de premio; "
    "sirve como herramienta de priorización creativa."
)
