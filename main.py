import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# ======================================
# Configuración de página y estilos
# ======================================
st.set_page_config(page_title="EDA + LLM (Coffee Sales)", page_icon="☕", layout="wide")
sns.set_theme(style="whitegrid")

# ======================================
# Utilidades LLM (Groq)
# ======================================
def ensure_groq() -> Groq:
    key = st.session_state.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("⚠️ Ingresa tu GROQ_API_KEY en la barra lateral.")
        st.stop()
    return Groq(api_key=key)

def ask_groq(messages, model: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> str:
    client = ensure_groq()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# ======================================
# Helpers de EDA
# ======================================
EXPECTED_COLS = [
    "hour_of_day","cash_type","money","coffee_name","Time_of_Day",
    "Weekday","Month_name","Weekdaysort","Monthsort","Date","Time"
]

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    msg = []
    if df.shape[0] < 300:
        msg.append(f"- El dataset tiene {df.shape[0]} filas (se recomiendan ≥ 300).")
    if df.shape[1] < 6:
        msg.append(f"- El dataset tiene {df.shape[1]} columnas (se recomiendan ≥ 6).")
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        msg.append(f"- Faltan columnas esperadas: {missing}")
    return (len(msg) == 0, "\n".join(msg))

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["hour_of_day","money","Weekdaysort","Monthsort"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Date" in out.columns:
        # Parsea y normaliza (mantén hora para futuros usos; para la serie usaremos .dt.normalize())
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", infer_datetime_format=True)
    return out

def iqr_outlier_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series([False]*len(series), index=series.index)
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    low, high = q1 - k*iqr, q3 + k*iqr
    return (series < low) | (series > high)

# ======================================
# Contexto para el LLM (basado SIEMPRE en TODO el dataset)
# ======================================
def build_llm_context(df: pd.DataFrame, max_cat_levels: int = 15) -> Dict[str, Any]:
    """
    Construye un contexto compacto (JSON) para el LLM usando SIEMPRE el dataset completo.
    Incluye: esquema, nulos, stats numéricas (excluyendo Weekdaysort/Monthsort donde aplique),
    top de categóricas y agregados cruzados por money (ingresos) y count (cantidad).
    """
    context: Dict[str, Any] = {}
    context["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    context["schema"] = {c: str(df[c].dtype) for c in df.columns}
    context["nulls"] = df.isnull().sum().to_dict()

    # Stats numéricas (excluye Weekdaysort/Monthsort)
    num_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in ["Weekdaysort", "Monthsort"]
    ]
    if num_cols:
        desc = df[num_cols].describe().round(3).to_dict()
        context["numeric_describe"] = desc

    # Conteos top de categóricas
    cat_cols = [c for c in df.columns if c not in df.select_dtypes(include=np.number).columns]
    top_counts = {}
    for c in cat_cols:
        vc = df[c].astype(str).value_counts().head(max_cat_levels).to_dict()
        top_counts[c] = vc
    context["categorical_top_counts"] = top_counts

    # Agregados clave (money totales)
    context["aggregates"] = {}
    if {"hour_of_day","money"} <= set(df.columns):
        context["aggregates"]["money_by_hour"] = df.groupby("hour_of_day")["money"].sum().sort_index().round(3).to_dict()
    if {"Weekday","money"} <= set(df.columns):
        context["aggregates"]["money_by_weekday"] = df.groupby("Weekday")["money"].sum().round(3).to_dict()
    if {"Monthsort","Month_name","money"} <= set(df.columns):
        tmp = df.groupby(["Monthsort","Month_name"])["money"].sum().reset_index().sort_values("Monthsort")
        context["aggregates"]["money_by_month"] = dict(zip(tmp["Month_name"], tmp["money"].round(3)))
    if {"cash_type","money"} <= set(df.columns):
        money_by_cash = df.groupby("cash_type")["money"].sum()
        total = float(money_by_cash.sum()) if money_by_cash.size else 0.0
        share = (money_by_cash / total * 100).round(2) if total else money_by_cash
        context["aggregates"]["money_by_cash_total"] = money_by_cash.round(3).to_dict()
        context["aggregates"]["money_by_cash_share"] = share.to_dict()

    # Agregados cruzados por money y count
    if {"coffee_name","Month_name"} <= set(df.columns):
        cross_money = df.groupby(["Month_name","coffee_name"])["money"].sum().reset_index()
        cross_count = df.groupby(["Month_name","coffee_name"]).size().reset_index(name="count")
        context["aggregates"]["money_by_coffee_and_month"] = {
            m: dict(zip(sub["coffee_name"], sub["money"].round(3)))
            for m, sub in cross_money.groupby("Month_name")
        }
        context["aggregates"]["count_by_coffee_and_month"] = {
            m: dict(zip(sub["coffee_name"], sub["count"]))
            for m, sub in cross_count.groupby("Month_name")
        }

    if {"coffee_name","Weekday"} <= set(df.columns):
        cross_money = df.groupby(["Weekday","coffee_name"])["money"].sum().reset_index()
        cross_count = df.groupby(["Weekday","coffee_name"]).size().reset_index(name="count")
        context["aggregates"]["money_by_coffee_and_weekday"] = {
            d: dict(zip(sub["coffee_name"], sub["money"].round(3)))
            for d, sub in cross_money.groupby("Weekday")
        }
        context["aggregates"]["count_by_coffee_and_weekday"] = {
            d: dict(zip(sub["coffee_name"], sub["count"]))
            for d, sub in cross_count.groupby("Weekday")
        }

    if {"coffee_name","hour_of_day"} <= set(df.columns):
        cross_money = df.groupby(["hour_of_day","coffee_name"])["money"].sum().reset_index()
        cross_count = df.groupby(["hour_of_day","coffee_name"]).size().reset_index(name="count")
        context["aggregates"]["money_by_coffee_and_hour"] = {
            int(h): dict(zip(sub["coffee_name"], sub["money"].round(3)))
            for h, sub in cross_money.groupby("hour_of_day")
        }
        context["aggregates"]["count_by_coffee_and_hour"] = {
            int(h): dict(zip(sub["coffee_name"], sub["count"]))
            for h, sub in cross_count.groupby("hour_of_day")
        }

    return context

def answer_with_context(question: str, context: Dict[str, Any], model: str, temperature: float = 0.2) -> str:
    """
    Responde SOLO usando el contexto (EDA + agregados del dataset completo).
    Soporta preguntas cruzadas por 'money' (ingresos) y 'count' (cantidad).
    Si falta info, debe decirlo claramente.
    """
    system = (
        "Eres un analista de datos. Responde preguntas SOLO usando el contexto JSON proporcionado. "
        "El contexto incluye agregados tanto por 'money' (ingresos) como por 'count' (cantidad de transacciones/unidades). "
        "Usa 'count' cuando la pregunta se refiera a cantidad/número de ventas/veces/unidades; "
        "usa 'money' cuando se refiera a ingresos/dinero/ganancias. "
        "Para preguntas por mes usa '..._by_coffee_and_month'; por día '..._by_coffee_and_weekday'; por hora '..._by_coffee_and_hour'. "
        "No digas 'no está explícito' si los datos existen en esas tablas; respóndelo directo. "
        "Si realmente no hay datos suficientes, dilo y sugiere qué faltaría."
    )
    user = f"""CONTEXTO:
{context}

PREGUNTA:
{question}

INSTRUCCIONES DE RESPUESTA:
- Responde en español, breve y claro, con el dato más relevante.
- Cita la métrica usada (money o count) si aplica.
- Incluye el valor exacto y la categoría ganadora.
- Si no hay datos suficientes, dilo explícitamente.
"""
    return ask_groq(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        model=model,
        temperature=temperature
    )

# ======================================
# UI — Barra lateral
# ======================================
with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input("GROQ_API_KEY", type="password")
    if api_key:
        st.session_state["GROQ_API_KEY"] = api_key

    st.header("🤖 Modelo Groq")
    model = st.selectbox("Modelo", ["llama-3.1-8b-instant","llama-3.1-70b-versatile"], index=0)
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.2, 0.05,
        help="Más bajo = más determinista; más alto = más creativo."
    )

# ======================================
# UI — Contenido principal
# ======================================
st.title("☕ EDA + LLM sobre Ventas de Café")
st.caption("Sube un CSV (≥300 filas, ≥6 columnas). Los filtros afectan EDA y visualizaciones; el LLM usa SIEMPRE el dataset completo. El gráfico de ingresos por mes no se filtra.")

uploaded = st.file_uploader("📂 Sube tu CSV (obligatorio)", type=["csv"])
if uploaded is None:
    st.warning("⚠️ Debes subir un archivo CSV para continuar.")
    st.stop()

# Carga y validación
df_full = pd.read_csv(uploaded)
df_full = cast_types(df_full)
ok, warn = validate_dataset(df_full)
if not ok:
    st.warning(f"Recomendaciones/Advertencias:\n{warn}")

# ========= Filtros (afectan SOLO EDA/visualizaciones; el LLM usa df_full) =========
available_months = []
if "Monthsort" in df_full.columns and "Month_name" in df_full.columns:
    tmp = df_full[["Monthsort","Month_name"]].dropna().drop_duplicates().sort_values("Monthsort")
    available_months = tmp["Month_name"].tolist()
elif "Month_name" in df_full.columns:
    available_months = sorted(df_full["Month_name"].dropna().unique().tolist())

available_coffees = []
if "coffee_name" in df_full.columns:
    available_coffees = ["Todos"] + sorted(df_full["coffee_name"].dropna().astype(str).unique().tolist())
else:
    available_coffees = ["Todos"]

col_filters = st.columns(3)
with col_filters[0]:
    sel_month = st.selectbox("🗓️ Mes (para EDA/visualizaciones)", options=["Todos"] + available_months if available_months else ["Todos"], index=0)
with col_filters[1]:
    sel_coffee = st.selectbox("☕ Tipo de café (para EDA/visualizaciones)", options=available_coffees, index=0)

# Aplica filtros SOLO para EDA/visualizaciones
df_view = df_full.copy()
if sel_month != "Todos" and "Month_name" in df_view.columns:
    df_view = df_view[df_view["Month_name"] == sel_month].copy()
if sel_coffee != "Todos" and "coffee_name" in df_view.columns:
    df_view = df_view[df_view["coffee_name"].astype(str) == sel_coffee].copy()

# =======================
# Vista rápida (df_view)
# =======================
st.subheader("👀 Vista previa")
st.dataframe(df_view.head())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Filas (vista)", df_view.shape[0])
with c2:
    st.metric("Columnas", df_view.shape[1])
with c3:
    # Excluimos Weekdaysort y Monthsort de numéricas analizadas
    num_cols = [
        c for c in df_view.select_dtypes(include=np.number).columns
        if c not in ["Weekdaysort", "Monthsort"]
    ]
    st.metric("Cols numéricas analizadas", len(num_cols))

st.subheader("🧱 Tipos de datos")
st.write(df_view.dtypes)

# =======================
# Nulos y descriptivos (df_view)
# =======================
st.subheader("🔎 Valores nulos por columna")
st.write(df_view.isnull().sum())  # solo tabla (sin gráfica)

st.subheader("📈 Estadísticas descriptivas (numéricas)")
if num_cols:
    st.write(df_view[num_cols].describe().T)
else:
    st.info("No se detectaron columnas numéricas en la vista actual.")

# =======================
# Outliers (IQR) en money (df_view)
# =======================
if "money" in df_view.columns:
    st.subheader("🚩 Atípicos (IQR) en 'money' (vista actual)")
    mask_out = iqr_outlier_mask(df_view["money"])
    st.write(f"Filas atípicas en 'money': {int(mask_out.sum())}")
    if mask_out.sum() > 0:
        st.dataframe(df_view.loc[mask_out].head())

# =======================
# Visualizaciones (df_view) — salvo ingresos por mes (df_full)
# =======================
st.subheader("📊 Visualizaciones")

viz1, viz2 = st.columns(2)

with viz1:
    if "money" in df_view.columns:
        st.markdown("**Distribución de 'money' (vista actual)**")
        fig, ax = plt.subplots()
        sns.histplot(df_view["money"].dropna(), kde=True, ax=ax)
        ax.set_xlabel("money")
        st.pyplot(fig)

    if {"coffee_name"} <= set(df_view.columns):
        st.markdown("**Top cafés vendidos (conteo, vista actual)**")
        top_c = df_view["coffee_name"].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_c.values, y=top_c.index, ax=ax)
        ax.set_xlabel("Cantidad")
        ax.set_ylabel("Café")
        st.pyplot(fig)

with viz2:
    if {"cash_type","money"} <= set(df_view.columns):
        st.markdown("**'money' por 'cash_type' (boxplot, vista actual)**")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_view, x="cash_type", y="money", ax=ax)
        ax.set_xlabel("cash_type")
        ax.set_ylabel("money")
        st.pyplot(fig)

    if {"hour_of_day","money"} <= set(df_view.columns):
        st.markdown("**Ingresos por hora (suma, vista actual)**")
        by_hour = df_view.groupby("hour_of_day")["money"].sum().reset_index().sort_values("hour_of_day")
        fig, ax = plt.subplots()
        sns.lineplot(data=by_hour, x="hour_of_day", y="money", marker="o", ax=ax)
        ax.set_xlabel("hour_of_day")
        ax.set_ylabel("money (suma)")
        st.pyplot(fig)

# 🔷 Evolución temporal por Date (vista actual)
if "Date" in df_view.columns and "money" in df_view.columns:
    st.markdown("**Evolución diaria de ingresos por `Date` (vista actual)**")
    # Normaliza a fecha (ignora horas) y suma
    ts = (df_view
          .dropna(subset=["Date"])
          .assign(Date_day=lambda d: d["Date"].dt.normalize())
          .groupby("Date_day")["money"].sum()
          .reset_index()
          .sort_values("Date_day"))
    if not ts.empty:
        fig, ax = plt.subplots()
        sns.lineplot(data=ts, x="Date_day", y="money", marker="o", ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("money (suma)")
        st.pyplot(fig)
    else:
        st.info("No hay datos con `Date` válidos en la vista actual para graficar.")

# Heatmap (df_view)
if {"Weekdaysort","hour_of_day","money","Weekday"} <= set(df_view.columns):
    st.markdown("**Heatmap: Suma de 'money' por (día × hora) — vista actual**")
    pivot = df_view.pivot_table(index="Weekdaysort", columns="hour_of_day", values="money", aggfunc="sum")
    day_names = (df_view[["Weekdaysort","Weekday"]]
                 .dropna()
                 .drop_duplicates()
                 .sort_values("Weekdaysort"))
    if not day_names.empty:
        pivot.index = [day_names.set_index("Weekdaysort").loc[i, "Weekday"]
                       if i in day_names["Weekdaysort"].values else i
                       for i in pivot.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="Blues", ax=ax)
    ax.set_xlabel("hour_of_day")
    ax.set_ylabel("Weekday")
    st.pyplot(fig)

# Barras por mes (NO varía con el filtro → usa df_full)
if {"Monthsort","Month_name","money"} <= set(df_full.columns):
    st.markdown("**Ingresos por mes (suma, TODO el período)**")
    by_month_full = (df_full.groupby(["Monthsort","Month_name"])["money"]
                     .sum().reset_index().sort_values("Monthsort"))
    fig, ax = plt.subplots()
    sns.barplot(data=by_month_full, x="Month_name", y="money", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("money (suma)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =======================
# Q&A BASADO EN DATOS (usa SIEMPRE df_full para el contexto)
# =======================
st.subheader("💬 Pregunta al agente")

# Construye el contexto SIEMPRE con df_full (no con df_view, así el LLM no varía por mes/café)
llm_context_full = build_llm_context(df_full, max_cat_levels=15)

user_q = st.text_input(
    "Escribe tu pregunta (ej.: ¿Cuál fue el café más vendido en octubre por cantidad? ¿Qué café deja más ingresos los lunes?)"
)
if st.button("Preguntar al LLM"):
    if not user_q.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        with st.spinner("Groq pensando…"):
            ans = answer_with_context(user_q, llm_context_full, model=model, temperature=temperature)
        st.markdown("**Respuesta del LLM:**")
        st.write(ans)

