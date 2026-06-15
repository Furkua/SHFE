import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Superhuman Formulation Engine", page_icon="💪", layout="wide")

# -----------------------------
# Demo data
# -----------------------------

def default_ingredients():
    # Valores nutricionales demo por 100 g. Reemplazar por datos reales cuando corresponda.
    rows = [
        ["Pasta de maní", "grasa/proteína/sabor", 590, 25, 20, 7, 50, 8, 7, 120, 4500, 35, 5],
        ["Maní", "crocante/proteína", 585, 26, 16, 5, 49, 7, 8, 80, 4200, 30, 4],
        ["Avena", "estructura/fibra", 389, 17, 66, 1, 7, 11, 10, 2, 1500, 35, 4],
        ["Avena polvo", "estructura/fibra", 389, 17, 66, 1, 7, 11, 10, 2, 1700, 20, 4],
        ["Chips choco", "sabor/dulzor", 500, 5, 60, 45, 30, 4, 2, 30, 5500, 25, 5],
        ["Maltitol", "endulzante", 240, 0, 98, 0, 0, 0, 0, 0, 3200, 35, 3],
        ["Almendras", "crocante/grasa/proteína", 579, 21, 22, 4, 50, 12, 12, 1, 9000, 35, 5],
        ["Maravilla", "semilla/textura", 584, 21, 20, 3, 51, 4, 9, 9, 2800, 20, 4],
        ["Linaza", "fibra/omega/textura", 534, 18, 29, 2, 42, 4, 27, 30, 3000, 15, 4],
        ["Albúmina", "proteína/estructura", 370, 80, 5, 1, 0, 0, 0, 1200, 12000, 12, 4],
        ["Clara", "proteína/estructura", 370, 80, 5, 1, 0, 0, 0, 1200, 11000, 12, 4],
        ["Panela", "dulzor", 383, 0, 98, 95, 0, 0, 0, 20, 2500, 8, 2],
        ["Chia", "fibra/omega/textura", 486, 17, 42, 1, 31, 3, 34, 16, 5500, 12, 4],
        ["Sorbato de potasio", "conservante", 0, 0, 0, 0, 0, 0, 0, 0, 9000, 1, 1],
        ["Sal de mar", "sabor", 0, 0, 0, 0, 0, 0, 0, 38700, 1200, 2, 1],
        ["Canela", "sabor", 247, 4, 81, 2, 1, 0, 53, 10, 7000, 2, 3],
        ["Lecitina", "emulsionante", 763, 0, 8, 4, 100, 15, 0, 0, 8000, 2, 4],
        ["Lecitina de soya", "emulsionante", 763, 0, 8, 4, 100, 15, 0, 0, 8000, 2, 4],
        ["Agua", "humedad/proceso", 0, 0, 0, 0, 0, 0, 0, 0, 2, 10, 1],
        ["Aceite de coco", "grasa/textura", 892, 0, 0, 0, 100, 87, 0, 0, 5500, 12, 3],
        ["Nueces", "grasa/crocante", 654, 15, 14, 3, 65, 6, 7, 2, 9500, 35, 5],
        ["Zapallo", "semilla/textura", 559, 30, 11, 1, 49, 9, 6, 7, 6000, 25, 5],
        ["Cranberry", "fruta/dulzor", 325, 0, 82, 65, 1, 0, 5, 5, 8000, 20, 4],
        ["Goji", "fruta/funcional", 349, 14, 77, 45, 1, 0, 13, 298, 12000, 15, 5],
    ]
    cols = ["Ingrediente", "Función", "kcal_100g", "proteina_100g", "carbos_100g", "azucar_100g", "grasa_100g", "sat_100g", "fibra_100g", "sodio_mg_100g", "costo_clp_kg", "max_pct", "score_funcional"]
    return pd.DataFrame(rows, columns=cols)

formula_choco_pct = {
    "Pasta de maní":16.2,"Maní":6.7,"Avena":16.3,"Avena polvo":1.5,"Chips choco":16.1,"Maltitol":15.9,
    "Almendras":8.6,"Maravilla":6.7,"Linaza":2.9,"Albúmina":2.5,"Panela":2.5,"Chia":2.4,"Sorbato de potasio":0.1,
    "Sal de mar":0.1,"Canela":0.1,"Lecitina":0.03,"Agua":1.3,
}
formula_berry_weights = {
    "Agua":0.84,"Sorbato de potasio":0.32,"Maltitol":32.07,"Aceite de coco":0,"Lecitina de soya":0.12,"Clara":5.35,
    "Nueces":23.79,"Almendras":39.35,"Zapallo":13.12,"Maravilla":10.39,"Cranberry":17.73,"Goji":8.71,"Chia":8.71,
    "Linaza":8.71,"Avena":25.5,"Avena polvo":2.97,
}

def weights_to_pct(weights):
    total = sum(weights.values())
    return {k: (v / total * 100 if total else 0) for k, v in weights.items()}

# -----------------------------
# Core calculations
# -----------------------------

def normalize_formula(df):
    out = df.copy()
    out["%"] = pd.to_numeric(out["%"], errors="coerce").fillna(0).clip(lower=0)
    total = out["%"].sum()
    if total > 0:
        out["%"] = out["%"] / total * 100
    return out


def calculate_nutrition(formula_df, ingredients_df, bar_weight=45):
    f = normalize_formula(formula_df)
    merged = f.merge(ingredients_df, on="Ingrediente", how="left")
    numeric_cols = ["kcal_100g","proteina_100g","carbos_100g","azucar_100g","grasa_100g","sat_100g","fibra_100g","sodio_mg_100g","costo_clp_kg","score_funcional"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    per100 = {}
    for col in ["kcal_100g","proteina_100g","carbos_100g","azucar_100g","grasa_100g","sat_100g","fibra_100g","sodio_mg_100g"]:
        per100[col] = float((merged["%"] / 100 * merged[col]).sum())
    cost_per_bar = float(((merged["%"] / 100 * bar_weight) / 1000 * merged["costo_clp_kg"]).sum())
    functional_score = float((merged["%"] / 100 * merged["score_funcional"]).sum())

    perbar = {
        "kcal": per100["kcal_100g"] * bar_weight / 100,
        "proteina_g": per100["proteina_100g"] * bar_weight / 100,
        "carbos_g": per100["carbos_100g"] * bar_weight / 100,
        "azucar_g": per100["azucar_100g"] * bar_weight / 100,
        "grasa_g": per100["grasa_100g"] * bar_weight / 100,
        "sat_g": per100["sat_100g"] * bar_weight / 100,
        "fibra_g": per100["fibra_100g"] * bar_weight / 100,
        "sodio_mg": per100["sodio_mg_100g"] * bar_weight / 100,
        "costo_clp": cost_per_bar,
        "score_funcional": functional_score,
    }
    return perbar, merged


def score_formula(nut, targets, goals):
    score = 50
    details = []
    if goals.get("max_protein", False):
        pts = min(15, max(0, nut["proteina_g"] / max(targets["proteina_min_g"], 0.1) * 15))
        score += pts
        details.append(f"Proteína: +{pts:.1f}")
    if goals.get("low_sugar", False):
        pts = 15 if nut["azucar_g"] <= targets["azucar_max_g"] else max(0, 15 - (nut["azucar_g"] - targets["azucar_max_g"]) * 3)
        score += pts
        details.append(f"Azúcar: +{pts:.1f}")
    if goals.get("high_fiber", False):
        pts = min(10, max(0, nut["fibra_g"] / max(targets["fibra_min_g"], 0.1) * 10))
        score += pts
        details.append(f"Fibra: +{pts:.1f}")
    if goals.get("low_cost", False):
        pts = 10 if nut["costo_clp"] <= targets["costo_max_clp"] else max(0, 10 - (nut["costo_clp"] - targets["costo_max_clp"]) / 20)
        score += pts
        details.append(f"Costo: +{pts:.1f}")
    if goals.get("functional", False):
        pts = min(10, nut["score_funcional"] * 2)
        score += pts
        details.append(f"Funcionalidad: +{pts:.1f}")
    if goals.get("low_sat", False):
        pts = 10 if nut["sat_g"] <= targets["sat_max_g"] else max(0, 10 - (nut["sat_g"] - targets["sat_max_g"]) * 3)
        score += pts
        details.append(f"Saturadas: +{pts:.1f}")
    return min(100, round(score, 1)), details


def build_formula_df(pct_dict):
    return pd.DataFrame({"Ingrediente": list(pct_dict.keys()), "%": list(pct_dict.values())})


def optimize_formula(ingredients_df, allowed_ingredients, targets, goals, bar_weight=45, iterations=2500):
    pool = ingredients_df[ingredients_df["Ingrediente"].isin(allowed_ingredients)].copy()
    if len(pool) < 3:
        return pd.DataFrame(columns=["Ingrediente", "%"]), None, []

    max_pct = pd.to_numeric(pool["max_pct"], errors="coerce").fillna(10).clip(lower=1).to_numpy()
    names = pool["Ingrediente"].tolist()

    best = None
    best_score = -1
    best_details = []
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        n = rng.integers(5, min(11, len(names)+1))
        idx = rng.choice(len(names), size=n, replace=False)
        raw = rng.random(n)
        pct = raw / raw.sum() * 100
        # penaliza y corrige máximos simples
        selected_max = max_pct[idx]
        pct = np.minimum(pct, selected_max)
        if pct.sum() <= 0:
            continue
        pct = pct / pct.sum() * 100
        candidate = pd.DataFrame({"Ingrediente": [names[i] for i in idx], "%": pct})
        nut, _ = calculate_nutrition(candidate, ingredients_df, bar_weight)
        sc, details = score_formula(nut, targets, goals)
        # Penalizaciones duras para metas activas
        if goals.get("max_protein") and nut["proteina_g"] < targets["proteina_min_g"]: sc -= 10
        if goals.get("low_sugar") and nut["azucar_g"] > targets["azucar_max_g"]: sc -= 10
        if goals.get("high_fiber") and nut["fibra_g"] < targets["fibra_min_g"]: sc -= 5
        if goals.get("low_cost") and nut["costo_clp"] > targets["costo_max_clp"]: sc -= 5
        if goals.get("low_sat") and nut["sat_g"] > targets["sat_max_g"]: sc -= 5
        if sc > best_score:
            best_score = sc
            best = candidate
            best_details = details

    if best is None:
        return pd.DataFrame(columns=["Ingrediente", "%"]), None, []
    best = normalize_formula(best).sort_values("%", ascending=False).reset_index(drop=True)
    nut, _ = calculate_nutrition(best, ingredients_df, bar_weight)
    return best, nut, best_details

# -----------------------------
# Session state
# -----------------------------
if "ingredients" not in st.session_state:
    st.session_state.ingredients = default_ingredients()
if "formula" not in st.session_state:
    st.session_state.formula = build_formula_df(formula_choco_pct)

# -----------------------------
# Hero
# -----------------------------
col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    st.image("assets/logo.png", use_container_width=True)
with col_title:
    st.markdown("# Superhuman Formulation Engine")
    st.caption("Demo local para formular, calcular, optimizar y rankear barras funcionales.")

st.divider()

# Sidebar controls
with st.sidebar:
    st.header("Configuración")
    bar_weight = st.number_input("Peso barra (g)", min_value=10.0, max_value=100.0, value=45.0, step=1.0)
    mode = st.radio("Modo de trabajo", ["Fórmula → Tabla nutricional", "Objetivo nutricional → Fórmula"], index=0)

    st.subheader("Cargar fórmula demo")
    if st.button("Usar fórmula Choco (%)"):
        st.session_state.formula = build_formula_df(formula_choco_pct)
    if st.button("Usar fórmula Berry (peso convertido a %)"):
        st.session_state.formula = build_formula_df(weights_to_pct(formula_berry_weights))

# Main tabs
tab_ing, tab_formula, tab_opt, tab_score = st.tabs(["1. Ingredientes", "2. Fórmula / Tabla", "3. Optimización", "4. Scoring"])

with tab_ing:
    st.subheader("Módulo de ingredientes")
    st.write("Edita la base de ingredientes o agrega filas nuevas. Los valores están expresados por 100 g y son demostrativos.")
    edited_ingredients = st.data_editor(
        st.session_state.ingredients,
        num_rows="dynamic",
        use_container_width=True,
        key="ingredients_editor",
    )
    st.session_state.ingredients = edited_ingredients

with tab_formula:
    st.subheader("Módulo de fórmula y tabla nutricional")
    if mode == "Fórmula → Tabla nutricional":
        st.write("Edita los porcentajes. La app normaliza automáticamente a 100%.")
        formula = st.data_editor(
            st.session_state.formula,
            num_rows="dynamic",
            use_container_width=True,
            column_config={"Ingrediente": st.column_config.SelectboxColumn("Ingrediente", options=st.session_state.ingredients["Ingrediente"].tolist())},
            key="formula_editor",
        )
        st.session_state.formula = normalize_formula(formula)
        nut, merged = calculate_nutrition(st.session_state.formula, st.session_state.ingredients, bar_weight)
    else:
        st.write("Define una tabla objetivo y usa el motor de optimización en la pestaña 3 para sugerir una fórmula.")
        nut, merged = calculate_nutrition(st.session_state.formula, st.session_state.ingredients, bar_weight)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Proteína", f"{nut['proteina_g']:.1f} g")
    c2.metric("Azúcar", f"{nut['azucar_g']:.1f} g")
    c3.metric("Fibra", f"{nut['fibra_g']:.1f} g")
    c4.metric("Calorías", f"{nut['kcal']:.0f} kcal")
    c5.metric("Costo demo", f"${nut['costo_clp']:.0f}")

    st.markdown("#### Fórmula normalizada")
    st.dataframe(st.session_state.formula, use_container_width=True)

    st.markdown("#### Tabla nutricional estimada")
    nutrition_table = pd.DataFrame([
        ["Energía", f"{nut['kcal']:.0f}", "kcal"],
        ["Proteínas", f"{nut['proteina_g']:.1f}", "g"],
        ["Carbohidratos", f"{nut['carbos_g']:.1f}", "g"],
        ["Azúcares", f"{nut['azucar_g']:.1f}", "g"],
        ["Grasas", f"{nut['grasa_g']:.1f}", "g"],
        ["Grasas saturadas", f"{nut['sat_g']:.1f}", "g"],
        ["Fibra", f"{nut['fibra_g']:.1f}", "g"],
        ["Sodio", f"{nut['sodio_mg']:.0f}", "mg"],
        ["Costo demo", f"{nut['costo_clp']:.0f}", "CLP/barra"],
    ], columns=["Variable", "Valor por barra", "Unidad"])
    st.dataframe(nutrition_table, use_container_width=True, hide_index=True)

with tab_opt:
    st.subheader("Motor de optimización por logros")
    st.write("Selecciona objetivos opcionales. El motor hace una búsqueda simple y propone una fórmula candidata.")

    g1, g2, g3 = st.columns(3)
    with g1:
        max_protein = st.checkbox("Maximizar proteína", value=True)
        low_sugar = st.checkbox("Reducir azúcar", value=True)
    with g2:
        high_fiber = st.checkbox("Aumentar fibra", value=False)
        low_cost = st.checkbox("Reducir costo", value=True)
    with g3:
        functional = st.checkbox("Mejorar perfil funcional", value=False)
        low_sat = st.checkbox("Reducir grasas saturadas", value=False)

    t1, t2, t3, t4, t5 = st.columns(5)
    targets = {
        "proteina_min_g": t1.number_input("Proteína mínima (g)", 0.0, 30.0, 8.0, 0.5),
        "azucar_max_g": t2.number_input("Azúcar máxima (g)", 0.0, 30.0, 6.0, 0.5),
        "fibra_min_g": t3.number_input("Fibra mínima (g)", 0.0, 20.0, 4.0, 0.5),
        "costo_max_clp": t4.number_input("Costo máximo demo ($)", 0.0, 2000.0, 350.0, 10.0),
        "sat_max_g": t5.number_input("Sat. máxima (g)", 0.0, 20.0, 4.0, 0.5),
    }
    goals = {
        "max_protein": max_protein,
        "low_sugar": low_sugar,
        "high_fiber": high_fiber,
        "low_cost": low_cost,
        "functional": functional,
        "low_sat": low_sat,
    }

    allowed = st.multiselect("Ingredientes permitidos", st.session_state.ingredients["Ingrediente"].tolist(), default=st.session_state.ingredients["Ingrediente"].tolist())

    if st.button("Generar fórmula candidata"):
        best_formula, best_nut, details = optimize_formula(st.session_state.ingredients, allowed, targets, goals, bar_weight)
        if best_nut is None:
            st.error("No hay suficientes ingredientes permitidos para optimizar.")
        else:
            st.session_state.formula = best_formula
            st.success("Fórmula candidata generada y cargada en la pestaña Fórmula / Tabla.")
            st.dataframe(best_formula, use_container_width=True, hide_index=True)
            st.write("Resultado estimado:")
            st.json({k: round(v, 2) for k, v in best_nut.items()})

with tab_score:
    st.subheader("Scoring system simple")
    st.write("Puntaje sobre 100 basado en los objetivos activos de la pestaña de optimización.")
    # Defaults if user has not visited optimizer controls in current run
    default_targets = {"proteina_min_g": 8.0, "azucar_max_g": 6.0, "fibra_min_g": 4.0, "costo_max_clp": 350.0, "sat_max_g": 4.0}
    default_goals = {"max_protein": True, "low_sugar": True, "high_fiber": False, "low_cost": True, "functional": False, "low_sat": False}
    try:
        active_targets = targets
        active_goals = goals
    except NameError:
        active_targets = default_targets
        active_goals = default_goals

    nut, _ = calculate_nutrition(st.session_state.formula, st.session_state.ingredients, bar_weight)
    score, details = score_formula(nut, active_targets, active_goals)
    st.metric("Score total", f"{score}/100")
    st.progress(score / 100)
    st.markdown("#### Desglose")
    if details:
        for d in details:
            st.write("- " + d)
    else:
        st.info("Activa logros en la pestaña 3 para ver un desglose más específico.")

    st.markdown("#### Interpretación")
    if score >= 85:
        st.success("Fórmula candidata fuerte para pasar a prototipo físico.")
    elif score >= 70:
        st.warning("Fórmula interesante, pero requiere ajustes antes de prototipar.")
    else:
        st.error("Fórmula débil según los objetivos seleccionados.")
