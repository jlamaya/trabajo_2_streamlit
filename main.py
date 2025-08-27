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
st.set_page_config(page_title="EDA + LLM (Coffee Sales)", page_icon="📊", layout="wide")
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
# EDA Helpers
# ======================================
EXPECTED_COLS = [
    "hour_of_day","cash_type","money","coffee_name","Time_of_Day",
    "Weekday","Month_name","Weekdaysort","Monthsort","Date","Time"
]

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    msg = []
    if df.shape[0] < 300:
        msg.append(f"- El dataset tiene {df.shape[0]} filas (se recomendan ≥ 300).")
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

def compute_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """ Calcula insights clave para UI/LLM. """
    insights = {}

    if {"coffee_name","money"} <= set(df.columns):
        top_coffee_count = df["coffee_name"].value_counts().head(10)
        top_coffee_money = df.groupby("coffee_name")["money"].sum().sort_values(ascending=False).head(10)
        insights["top_coffee_count"] = top_coffee_count.to_dict()
        insights["top_coffee_money"] = top_coffee_money.to_dict()

    if {"hour_of_day","money"} <= set(df.columns):
        money_by_hour = df.groupby("hour_of_day")["money"].sum().sort_values(ascending=False)
        insights["best_hour_by_money"] = money_by_hour.index[0] if not money_by_hour.empty else None
        insights["money_by_hour"] = money_by_hour.sort_index().to_dict()

    if {"Weekday","money"} <= set(df.columns):
        money_by_day = df.groupby("Weekday")["money"].sum().sort_values(ascending=False)
        insights["best_weekday_by_money"] = money_by_day.index[0] if not money_by_day.empty else None
        insights["money_by_weekday"] = money_by_day.to_dict()

    if {"cash_type","money"} <= set(df.columns):
        money_by_cash = df.groupby("cash_type")["money"].sum().sort_values(ascending=False)
        total_money = float(money_by_cash.sum()) if money_by_cash.size else 0.0
        share = (money_by_cash / total_money * 100).round(2) if total_money else money_by_cash
        insights["cash_type_total"] = money_by_cash.to_dict()
        insights["cash_type_share"] = share.to_dict()

    if {"Monthsort","Month_name","money"} <= set(df.columns):
        tmp = df.groupby(["Monthsort","Month_name"])["money"].sum().reset_index()
        tmp = tmp.sort_values(["Monthsort","Month_name"])
        if not tmp.empty:
            best_row = tmp.loc[tmp["money"].idxmax()]
            insights["best_month"] = str(best_row["Month_name"])
            insights["money_by_month"] = dict(zip(tmp["Month_name"], tmp["money"]))

    # Rango de fechas si existe
    if "Date" in df.columns:
        insights["date_min"] = str(pd.to_datetime(df["Date"]).min())
        insights["date_max"] = str(pd.to_datetime(df["Date"]).max())

    return insights

def insights_to_bullets(ins: Dict[str, Any]) -> str:
    """ Resume dict de insights a texto legible. """
    lines = []
    if ins.get("best_hour_by_money") is not None:
        lines.append(f"- Hora con mayores ingresos: {int(ins['best_hour_by_money'])}.")
    if ins.get("best_weekday_by_money"):
        lines.append(f"- Día con mayores ingresos: {ins['best_weekday_by_money']}.")
    if ins.get("best_month"):
        lines.append(f"- Mes más fuerte en ingresos: {ins['best_month']}.")
    if "cash_type_share" in ins:
        parts = ", ".join([f"{k}: {v:.2f}%" for k, v in ins["cash_type_share"].items()])
        if parts:
            lines.append(f"- Participación por método de pago: {parts}.")
    if "top_coffee_count" in ins and ins["top_coffee_count"]:
        top = list(ins["top_coffee_count"].items())[:3]
        lines.append("- Cafés más vendidos (cantidad): " + ", ".join([f"{k} ({v})" for k, v in top]) + ".")
    if "top_coffee_money" in ins and ins["top_coffee_money"]:
        topm = list(ins["top_coffee_money"].items())[:3]
        lines.append("- Cafés con más ingresos: " + ", ".join([f"{k}" for k, _ in topm]) + ".")
    if ins.get("date_min") or ins.get("date_max"):
        lines.append(f"- Rango de fechas: {ins.get('date_min')} → {ins.get('date_max')}.")
    return "\n".join(lines) if lines else "(Sin insights calculables)"

def build_llm_context(df: pd.DataFrame, insights: Dict[str, Any], max_cat_levels: int = 15) -> Dict[str, Any]:
    """
    Construye un contexto compacto para el LLM basado en datos e insights.
    Incluye: esquema, nulos, stats numéricas, conteos top de categóricas,
    agregados clave y combinaciones relevantes.
    """
    context: Dict[str, Any] = {}
    context["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    context["schema"] = {c: str(df[c].dtype) for c in df.columns}
    context["nulls"] = df.isnull().sum().to_dict()

    # Stats numéricas
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(3).to_dict()
        context["numeric_describe"] = desc

    # Conteos top de categóricas
    cat_cols = [c for c in df.columns if c not in num_cols]
    top_counts = {}
    for c in cat_cols:
        vc = df[c].astype(str).value_counts().head(max_cat_levels).to_dict()
        top_counts[c] = vc
    context["categorical_top_counts"] = top_counts

    # Agregados clave
    context["aggregates"] = {}

    if {"hour_of_day","money"} <= set(df.columns):
        context["aggregates"]["money_by_hour"] = (
            df.groupby("hour_of_day")["money"].sum().sort_index().round(3).to_dict()
        )
    if {"Weekday","money"} <= set(df.columns):
        context["aggregates"]["money_by_weekday"] = (
            df.groupby("Weekday")["money"].sum().round(3).to_dict()
        )
    if {"Monthsort","Month_name","money"} <= set(df.columns):
        tmp = df.groupby(["Monthsort","Month_name"])["money"].sum().reset_index().sort_values("Monthsort")
        context["aggregates"]["money_by_month"] = dict(zip(tmp["Month_name"], tmp["money"].round(3)))
    if {"cash_type","money"} <= set(df.columns):
        money_by_cash = df.groupby("cash_type")["money"].sum()
        total = float(money_by_cash.sum()) if money_by_cash.size else 0.0
        share = (money_by_cash / total * 100).round(2) if total else money_by_cash
        context["aggregates"]["money_by_cash_total"] = money_by_cash.round(3).to_dict()
        context["aggregates"]["money_by_cash_share"] = share.to_dict()

    # 🔥 Agregados cruzados (para preguntas específicas)
    if {"coffee_name","Month_name"} <= set(df.columns):
        cross = df.groupby(["Month_name","coffee_name"])["money"].sum().reset_index()
        pivot = {}
        for month in cross["Month_name"].unique():
            subset = cross[cross["Month_name"] == month]
            pivot[month] = dict(zip(subset["coffee_name"], subset["money"].round(3)))
        context["aggregates"]["money_by_coffee_and_month"] = pivot

    if {"coffee_name","Weekday"} <= set(df.columns):
        cross = df.groupby(["Weekday","coffee_name"])["money"].sum().reset_index()
        pivot = {}
        for day in cross["Weekday"].unique():
            subset = cross[cross["Weekday"] == day]
            pivot[day] = dict(zip(subset["coffee_name"], subset["money"].round(3)))
        context["aggregates"]["money_by_coffee_and_weekday"] = pivot

    if {"coffee_name","hour_of_day"} <= set(df.columns):
        cross = df.groupby(["hour_of_day","coffee_name"])["money"].sum().reset_index()
        pivot = {}
        for hour in cross["hour_of_day"].unique():
            subset = cross[cross["hour_of_day"] == hour]
            pivot[int(hour)] = dict(zip(subset["coffee_name"], subset["money"].round(3)))
        context["aggregates"]["money_by_coffee_and_hour"] = pivot

    # Insights ya calculados
    context["insights"] = insights
    return context

def answer_with_context(question: str, context: Dict[str, Any], model: str, temperature: float = 0.2) -> str:
    """
    Responde usando SOLO el contexto. Si falta info, debe decirlo.
    """
    system = (
        "Eres un analista de datos. Debes responder SOLO usando el contexto proporcionado. "
        "Si el contexto no tiene la información necesaria, dilo explícitamente y explica qué faltaría. "
        "Cuando cites cifras, incluye unidades cuando sea obvio (por ejemplo, 'money' como monto). "
        "Si haces un cálculo, muéstralo brevemente."
    )
    user = f"""CONTEXTO (JSON compacto):
{context}

PREGUNTA:
{question}

INSTRUCCIONES DE RESPUESTA:
- Responde en español, breve y claro.
- No inventes datos más allá del contexto.
- Si la respuesta requiere datos que no están, dilo y sugiere qué cálculo o dato faltaría.
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
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                            help="Más bajo = respuestas más deterministas; más alto = más creativas.")

# ======================================
# UI — Contenido principal
# ======================================
st.title("📊 EDA + LLM sobre Ventas de Café")
st.caption("Sube un CSV (≥300 filas, ≥6 columnas). El EDA mostrará estructura, nulos, outliers y visualizaciones. Luego, conversa con un LLM usando los insights.")

uploaded = st.file_uploader("📂 Sube tu CSV (obligatorio)", type=["csv"])

if uploaded is None:
    st.warning("⚠️ Debes subir un archivo CSV para continuar.")
    st.stop()

# Carga y validación
df_raw = pd.read_csv(uploaded)
df = cast_types(df_raw)

ok, warn = validate_dataset(df)
if not ok:
    st.warning(f"Recomendaciones/Advertencias:\n{warn}")

# =======================
# Vista rápida
# =======================
st.subheader("👀 Vista previa")
st.dataframe(df.head())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Filas", df.shape[0])
with c2:
    st.metric("Columnas", df.shape[1])
with c3:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.metric("Cols numéricas", len(num_cols))

st.subheader("🧱 Tipos de datos")
st.write(df.dtypes)

# =======================
# Nulos y descriptivos
# =======================
st.subheader("🔎 Valores nulos por columna")
nulls = df.isnull().sum()
st.write(nulls)

fig, ax = plt.subplots()
nulls.plot(kind="bar", ax=ax)
ax.set_title("Conteo de nulos por columna")
ax.set_ylabel("Nulos")
st.pyplot(fig)

st.subheader("📈 Estadísticas descriptivas (numéricas)")
if num_cols:
    st.write(df[num_cols].describe().T)
else:
    st.info("No se detectaron columnas numéricas.")

# =======================
# Outliers (IQR) en money
# =======================
if "money" in df.columns:
    st.subheader("🚩 Detección de atípicos (IQR) en 'money'")
    mask_out = iqr_outlier_mask(df["money"])
    st.write(f"Filas atípicas en 'money': {int(mask_out.sum())}")
    if mask_out.sum() > 0:
        st.dataframe(df.loc[mask_out].head())

# =======================
# Visualizaciones
# =======================
st.subheader("📊 Visualizaciones")

viz1, viz2 = st.columns(2)

with viz1:
    if "money" in df.columns:
        st.markdown("**Distribución de 'money'**")
        fig, ax = plt.subplots()
        sns.histplot(df["money"].dropna(), kde=True, ax=ax)
        ax.set_xlabel("money")
        st.pyplot(fig)

    if {"coffee_name"} <= set(df.columns):
        st.markdown("**Top cafés vendidos (conteo)**")
        top_c = df["coffee_name"].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_c.values, y=top_c.index, ax=ax)
        ax.set_xlabel("Cantidad")
        ax.set_ylabel("Café")
        st.pyplot(fig)

with viz2:
    if {"cash_type","money"} <= set(df.columns):
        st.markdown("**'money' por 'cash_type' (boxplot)**")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="cash_type", y="money", ax=ax)
        ax.set_xlabel("cash_type")
        ax.set_ylabel("money")
        st.pyplot(fig)

    if {"hour_of_day","money"} <= set(df.columns):
        st.markdown("**Ingresos por hora (suma)**")
        by_hour = df.groupby("hour_of_day")["money"].sum().reset_index().sort_values("hour_of_day")
        fig, ax = plt.subplots()
        sns.lineplot(data=by_hour, x="hour_of_day", y="money", marker="o", ax=ax)
        ax.set_xlabel("hour_of_day")
        ax.set_ylabel("money (suma)")
        st.pyplot(fig)

# Heatmap hora x día
if {"Weekdaysort","hour_of_day","money","Weekday"} <= set(df.columns):
    st.markdown("**Heatmap: Suma de 'money' por (día de semana × hora)**")
    pivot = df.pivot_table(index="Weekdaysort", columns="hour_of_day", values="money", aggfunc="sum")
    # Etiquetas legibles si hay Weekday
    day_names = (df[["Weekdaysort","Weekday"]]
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

# Barras por mes
if {"Monthsort","Month_name","money"} <= set(df.columns):
    st.markdown("**Ingresos por mes (suma)**")
    by_month = df.groupby(["Monthsort","Month_name"])["money"].sum().reset_index().sort_values("Monthsort")
    fig, ax = plt.subplots()
    sns.barplot(data=by_month, x="Month_name", y="money", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("money (suma)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =======================
# Insights + LLM (resumen)
# =======================
st.subheader("🧠 Insights y LLM")

ins = compute_insights(df)
ins_bullets = insights_to_bullets(ins)
with st.expander("Ver insights calculados (para contexto del LLM)"):
    st.markdown(ins_bullets)

summary_prompt = f"""
Eres un analista de datos. Te doy la estructura básica de un dataset de ventas de café y algunos insights calculados.

Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas.
Tipos de datos (por columna): {df.dtypes.to_dict()}
Nulos por columna: {df.isnull().sum().to_dict()}

Insights calculados:
{ins_bullets}

Por favor, entrega:
1) Un resumen ejecutivo (3-5 viñetas) de patrones clave (horas, días, meses, cafés y método de pago).
2) Sugerencias de hipótesis o próximos análisis a realizar (2-3 puntos).
Responde en español, claro y sin inventar datos que no estén respaldados por lo anterior.
"""

if st.button("📄 Generar resumen con LLM"):
    with st.spinner("Consultando Groq…"):
        resumen = ask_groq(
            messages=[{"role": "user", "content": summary_prompt}],
            model=model,
            temperature=temperature
        )
    st.markdown("### Resumen (LLM)")
    st.write(resumen)
    st.session_state["eda_summary"] = resumen

# =======================
# Q&A BASADO EN DATOS (CON CONTEXTO)
# =======================
st.subheader("💬 Pregunta al agente (basado en datos e insights)")

# Construimos el contexto SIEMPRE después del EDA
llm_context = build_llm_context(df, insights=ins, max_cat_levels=15)

with st.expander("Ver contexto que recibe el LLM (JSON compacto)"):
    st.json(llm_context)

user_q = st.text_input(
    "Escribe tu pregunta (ej.: ¿Qué café conviene promocionar en la mañana? ¿Qué día y hora dejan más ingresos?)"
)
if st.button("Preguntar al LLM con contexto"):
    if not user_q.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        with st.spinner("Groq pensando…"):
            ans = answer_with_context(user_q, llm_context, model=model, temperature=temperature)
        st.markdown("**Respuesta del LLM (basada SOLO en el contexto):**")
        st.write(ans)
