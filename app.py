import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Proyecto Final Python for Analytics - Análisis Exploratorio de Datos",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def info_general(self):
        info = pd.DataFrame({
            'Columna': self.df.columns,
            'Tipo de Dato': self.df.dtypes.values,
            'No Nulos': self.df.count().values,
            'Nulos': self.df.isnull().sum().values,
            '% Nulos': (self.df.isnull().sum().values / len(self.df)) * 100
        })
        return info
    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = self.df.select_dtypes(include=["object"]).columns.tolist()
        return numericas, categoricas
    def estadisticas_descriptivas(self):
        return self.df.describe()
    def conteo_nulos(self):
        nulos = self.df.isnull().sum()
        pct = (nulos / len(self.df) * 100).round(2)
        return pd.DataFrame({"Nulos": nulos, "Porcentaje (%)": pct})[
            (nulos > 0)
        ]
    def histograma(self, col: str, hue: str = None):
        fig, ax = plt.subplots(figsize=(7, 4))
        if hue and hue in self.df.columns:
            for val in self.df[hue].dropna().unique():
                subset = self.df[self.df[hue] == val][col].dropna()
                ax.hist(subset, bins=30, alpha=0.6, label=str(val))
            ax.legend(title=hue)
        else:
            ax.hist(self.df[col].dropna(), bins=30, color="#3776AB", alpha=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de {col}")
        plt.tight_layout()
        return fig
    def barras_categorica(self, col: str):
        counts = self.df[col].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
        ax.set_xlabel(col)
        ax.set_ylabel("Cantidad")
        ax.set_title(f"Distribución de {col}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig
    def boxplot_num_cat(self, num_col: str, cat_col: str):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, ax=ax, palette="Set2")
        ax.set_title(f"{num_col} por {cat_col}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig
    def heatmap_cat_cat(self, col1: str, col2: str):
        ct = pd.crosstab(self.df[col1], self.df[col2], normalize="index").round(3)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(ct, annot=True, fmt=".1%", cmap="Blues", ax=ax)
        ax.set_title(f"{col1} vs {col2} (proporciones por fila)")
        plt.tight_layout()
        return fig
    
st.sidebar.image("logo1.png", width="stretch")
st.sidebar.title("📊 Menú")
menu = st.sidebar.selectbox(
    "Seleccione módulo",
    ["🏠 Home", "📂 Carga de Dataset", "📊 Análisis Exploratorio de Datos", "🎯 Conclusiones"]
) 
st.markdown(
        "<small>Especialización Python for Analytics · 2026</small>",
        unsafe_allow_html=True,
    )
# =================
#   MoDULO 1 HOME
# =================
if menu == "🏠 Home":
    st.markdown(
        """
        <style>
        .stElementContainer h1 {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("📊 Proyecto Final Python for Analytics - Análisis Exploratorio de Datos")
    st.divider()
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            #### 🎯 Objetivo del proyecto
            Este proyecto aplica técnicas de **Análisis Exploratorio de Datos (EDA)**
            sobre el dataset **TelcoCustomerChurn.csv** para identificar patrones
            asociados a la fuga de clientes en una empresa de telecomunicaciones.

            Este análisis **no** tiene como objetivo construir modelos predictivos,
            sino **comprender el comportamiento de los clientes** mediante
            estadística descriptiva y visualizaciones interactivas.

            #### 🗂️ Sobre el dataset
            El archivo contiene información de **7 043 clientes** con 21 variables que
            describen datos demográficos, servicios contratados, facturación mensual,
            tiempo de permanencia y si el cliente abandonó la empresa (`Churn`).
            """,
        )
        st.info("""
        Durante el último año, agravado por el contexto COVID-19, la tasa de
        churn subió **+0.5 pp** (de 2 % a 2.5 %). Dado que adquirir un nuevo
        cliente cuesta entre **6 y 7 veces** más que retener uno existente,
        comprender las causas de la fuga es sumamente crítico.
        """)

        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.metric(
                label="Tasa de Churn (Anual)", 
                value="2.5 %", 
                delta="0.5 pp", 
                delta_color="inverse"
            )

        with sub_col2:
            st.metric(
                label="Costo de Adquisición", 
                value="7x", 
                delta="vs. Retención"
            )
    
    with col2:
        st.markdown(
            """
            #### 👩‍💻 Autora
            
            **Nombre completo:** Gianella Sophia Alarcón Bardales

            **Especialización:** Python for Analytics

            **Año:** 2026
            """
        )
        st.divider()
        st.markdown("""
        <style>
        .badge-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .badge {
            padding: 4px 12px;
            font-size: 13px;
            font-weight: 600;
            color: white;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("#### 🛠️ Tecnologías empleadas")

        tecnologias = {
                "Streamlit": "#FF4B4B",
                "Python": "#3776AB",
                "Pandas": "#150458",
                "NumPy": "#4dabcf",
                "Matplotlib": "#11557c",
                "Seaborn": "#34b4eb"
            }

        pills_html = '<div class="badge-container">'
        for tech, color in tecnologias.items():
            pills_html += f'<span class="badge" style="background-color: {color};">{tech}</span>'
        pills_html += '</div>'
            
        st.markdown(pills_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📌 Estructura de la aplicación")
    st.markdown(
    """
    | Módulo | Contenido Principal |
    | :--- | :--- |
    | 🏠 **Home** | Introducción, objetivos y contexto de negocio. |
    | 📂 **Carga del Dataset** | Limpieza de datos y validación de datos. |
    | 🔍 **EDA** | Exploración de 10 hallazgos clave sobre la fuga de clientes. |
    | 📝 **Conclusiones** | Recomendaciones estratégicas e insights finales. |
    """
    )
    st.divider()
# ===========================
#  MoDULO 2 CARGA DE DATASET
# ===========================
elif menu == "📂 Carga de Dataset":
    st.title("📂 Carga del Dataset")
    uploaded_file = st.file_uploader("**Cargar archivo CSV**", type=["csv"]) 

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        st.session_state['df'] = df
        st.success("Archivo cargado con éxito")
        
        st.subheader("Vista previa del Dataset")
        st.dataframe(df.head(10)) 
        
        st.subheader("Dimensiones del Dataset")
        col1, col2 = st.columns(2)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        
    else:
        st.warning("Por favor, sube un archivo para continuar.")
        st.stop()
    st.divider()
# ===========================
#  MoDULO 3 EDA
# ===========================
elif menu == "📊 Análisis Exploratorio de Datos":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    
    if "df" not in st.session_state:
        st.warning("Primero carga el dataset en el módulo **📂 Carga de Dataset**.")
        st.stop()

    df = st.session_state["df"]
    analyzer = DataAnalyzer(df)
    numericas, categoricas = analyzer.clasificar_variables()

    tabs = st.tabs([
        "1 · Info General",
        "2 · Variables",
        "3 · Estadísticas",
        "4 · Valores faltantes",
        "5 · Variables numéricas",
        "6 · Variables categóricas",
        "7 · Bivariado N vs. C",
        "8 · Bivariado C vs. C",
        "9 · Análisis Dinámico",
        "10 · Hallazgos clave",
    ])
    with tabs[0]:
        st.header("Información General del Dataset")
        st.markdown("Análisis de tipos de datos y valores nulos por columna.")
        info_df = analyzer.info_general()
        st.dataframe(info_df, width="stretch")
        
        with st.expander("Ver interpretación"):
            st.markdown("""
            - La columna `TotalCharges` tiene valores nulos porque fue cargada como texto y convertida a número.
            - Las columnas como `SeniorCitizen` son numéricas (0/1) pero representan una categoría.
            """)
    with tabs[1]:
        st.header("Clasificación de Variables")
        num_vars, cat_vars = analyzer.clasificar_variables()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔢 Variables Numéricas", len(num_vars))

            with st.expander("Ver variables numéricas"):
                num_cols = st.columns(3)
                for i, var in enumerate(num_vars):
                    num_cols[i % 3].markdown(f"• `{var}`")
        
        with col2:
            st.metric("🏷️ Variables Categóricas", len(cat_vars))
            with st.expander("Ver variables categóricas"):
                cat_cols = st.columns(3)
                for i, var in enumerate(cat_vars):
                    cat_cols[i % 3].markdown(f"• `{var}`")
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Numéricas", "Categóricas"], [len(numericas), len(categoricas)],
               color=["#3776AB", "#FF4B4B"])
        ax.set_ylabel("Cantidad de variables")
        ax.set_title("Distribución de tipos de variables")
        plt.tight_layout()
        st.pyplot(fig)
        
    with tabs[2]:
        st.header("Estadísticas Descriptivas")
        st.markdown("""
        Resumen estadístico de cada variable numérica:
        media, mediana (percentil 50), desviación estándar, mínimo, máximo y cuartiles.
        """)

        desc = analyzer.estadisticas_descriptivas()
        st.dataframe(desc.style.format("{:.2f}"), width="stretch")

        st.markdown("##### Interpretación rápida")

        col1, col2, col3 = st.columns(3)
        col1.metric("Media MonthlyCharges", f"${df['MonthlyCharges'].mean():.2f}")
        col2.metric("Mediana tenure", f"{df['tenure'].median():.0f} meses")
        col3.metric("Media TotalCharges", f"${df['TotalCharges'].mean():.2f}")

        st.info("""
        - **MonthlyCharges** tiene alta dispersión (std ≈ 30), lo que indica una base de 
          clientes con perfiles de gasto muy variados.
        - **tenure** presenta distribución bimodal: clientes muy nuevos (~1 mes) y clientes 
          muy leales (~72 meses), sin un centro claro.
        - **TotalCharges** tiene alta asimetría positiva, influenciada por clientes con 
          largo tiempo de permanencia.
        """)
    with tabs[3]:
        st.header("Análisis de valores faltantes")

        nulos_df = analyzer.conteo_nulos()

        if nulos_df.empty:
            st.success("✅ No se encontraron valores nulos en el dataset.")
        else:
            st.dataframe(nulos_df, width="stretch")

        st.markdown("""
        Los valores nulos en `TotalCharges` corresponden a clientes
        con **tenure = 0**(nuevos clientes que aún no han pagado), es decir, clientes que no llegaron a completar su primer mes.
        No requiere imputación, pero se debe tener en cuenta en análisis posteriores.
        """)

        nulos_count = df.isnull().sum().sum()
        col1, col2 = st.columns(2)
        col1.metric("Total valores nulos", int(nulos_count))
        col2.metric("% sobre el total de datos", f"{nulos_count / df.size * 100:.3f}%")

        if not nulos_df.empty:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(nulos_df.index, nulos_df["Nulos"], color="#FF4B4B")
            ax.set_ylabel("Cantidad de nulos")
            ax.set_title("Columnas con valores faltantes")
            plt.tight_layout()
            st.pyplot(fig)
    with tabs[4]:
        st.header("Distribución de variables numéricas")

        col_sel = st.selectbox("Selecciona variable numérica", numericas, key="dist_num")
        mostrar_churn = st.checkbox("Segmentar por Churn", key="chk_churn_dist")

        hue = "Churn" if mostrar_churn else None
        fig = analyzer.histograma(col_sel, hue=hue)
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        serie = df[col_sel].dropna()
        col1.metric("Media", f"{serie.mean():.2f}")
        col2.metric("Mediana", f"{serie.median():.2f}")
        col3.metric("Desv. Estándar", f"{serie.std():.2f}")

        st.markdown(f"""
        **Interpretación:** La variable `{col_sel}` presenta una media de **{serie.mean():.2f}**
        y una mediana de **{serie.median():.2f}**. {"La media > mediana sugiere asimetría positiva." 
        if serie.mean() > serie.median() else "La media < mediana sugiere asimetría negativa."}
        """)
    with tabs[5]:
        st.subheader("Análisis de variables categóricas")

        cat_sel = st.selectbox("Selecciona variable categórica", categoricas, key="cat_sel")
        fig = analyzer.barras_categorica(cat_sel)
        st.pyplot(fig)

        counts = df[cat_sel].value_counts()
        pcts = (counts / len(df) * 100).round(2)
        tabla = pd.DataFrame({"Conteo": counts, "Proporción (%)": pcts})
        st.dataframe(tabla, width="stretch")

        st.markdown(f"""
        **Interpretación:** La categoría más frecuente en `{cat_sel}` es 
        **'{counts.index[0]}'** con **{counts.iloc[0]:,}** registros 
        ({pcts.iloc[0]:.1f}% del total).
        """)

    with tabs[6]:
        st.header("Análisis Bivariado: Numérico vs Categórico")

        st.markdown("""
        Comparamos variables numéricas segmentadas por una variable categórica
        (típicamente `Churn`) para identificar diferencias entre grupos.
        """)

        col1, col2 = st.columns(2)
        with col1:
            num_biv = st.selectbox("Variable numérica", numericas, key="num_biv")
        with col2:
            cat_biv = st.selectbox("Variable categórica", categoricas, key="cat_biv",
                                   index=categoricas.index("Churn") if "Churn" in categoricas else 0)

        fig = analyzer.boxplot_num_cat(num_biv, cat_biv)
        st.pyplot(fig)

        st.markdown("##### Estadísticas por grupo")
        grupo_stats = df.groupby(cat_biv)[num_biv].agg(["mean", "median", "std"]).round(2)
        grupo_stats.columns = ["Media", "Mediana", "Desv. Estándar"]
        st.dataframe(grupo_stats, width="stretch")

        if cat_biv == "Churn" and num_biv in ["MonthlyCharges", "tenure"]:
            churn_yes = df[df["Churn"] == "Yes"][num_biv].mean()
            churn_no = df[df["Churn"] == "No"][num_biv].mean()
            diff = abs(churn_yes - churn_no)
            st.info(f"""
            Los clientes que se fueron (Churn=Yes) tienen una media de `{num_biv}` de **{churn_yes:.2f}**,
            mientras los que permanecen tienen **{churn_no:.2f}** — una diferencia de **{diff:.2f}**.
            """)
    with tabs[7]:
        st.header("Análisis Bivariado: Categórico vs Categórico")
        st.markdown("""
        Análisis de la relación entre dos variables categóricas mediante tablas de
        contingencia y mapas de calor con proporciones por fila.
        """)

        col1, col2 = st.columns(2)
        with col1:
            cat1 = st.selectbox("Variable categórica (eje Y)", categoricas, key="cat1",
                                index=categoricas.index("Contract") if "Contract" in categoricas else 0)
        with col2:
            cat2 = st.selectbox("Variable categórica (eje X)", categoricas, key="cat2",
                                index=categoricas.index("Churn") if "Churn" in categoricas else 1)

        fig = analyzer.heatmap_cat_cat(cat1, cat2)
        st.pyplot(fig)

        st.markdown("##### Tabla de contingencia (conteos absolutos)")
        ct_abs = pd.crosstab(df[cat1], df[cat2])
        st.dataframe(ct_abs, width="stretch")

        if cat1 == "Contract" and cat2 == "Churn":
            st.info("""
            **Insight clave:** Los clientes con contrato **Month-to-month** tienen la tasa de
            churn más alta. Los contratos anuales y bianuales muestran una retención
            significativamente mayor, lo que indica que los compromisos a largo plazo
            reducen la fuga.
            """)
    with tabs[8]:
        st.header("Análisis basado en parámetros seleccionado")
        st.markdown("Selecciona variables para explorar relaciones de forma dinámica.")

        tipo_analisis = st.selectbox(
            "Tipo de análisis",
            ["Distribución numérica", "Barras categórica", "Num vs Cat (boxplot)", "Cat vs Cat (heatmap)"],
            key="tipo_custom",
        )

        if tipo_analisis == "Distribución numérica":
            var = st.selectbox("Variable", numericas, key="cust_num")
            bins = st.slider("Número de bins", 5, 80, 30, key="cust_bins")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df[var].dropna(), bins=bins, color="#3776AB", alpha=0.85, edgecolor="white")
            ax.set_title(f"Distribución de {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig)

        elif tipo_analisis == "Barras categórica":
            var = st.selectbox("Variable", categoricas, key="cust_cat")
            top_n = st.slider("Top N categorías", 2, 20, 10, key="cust_topn")
            counts = df[var].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=counts.index, y=counts.values, palette="Blues_d", ax=ax)
            ax.set_title(f"Top {top_n} categorías de {var}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        elif tipo_analisis == "Num vs Cat (boxplot)":
            col1, col2 = st.columns(2)
            with col1:
                num_c = st.selectbox("Variable numérica", numericas, key="cust_nc")
            with col2:
                cat_c = st.selectbox("Variable categórica", categoricas, key="cust_cc")
            fig = analyzer.boxplot_num_cat(num_c, cat_c)
            st.pyplot(fig)

        elif tipo_analisis == "Cat vs Cat (heatmap)":
            col1, col2 = st.columns(2)
            cats_multi = st.multiselect(
                "Selecciona dos variables categóricas", categoricas, default=categoricas[:2], key="cust_multi"
            )
            if len(cats_multi) == 2:
                fig = analyzer.heatmap_cat_cat(cats_multi[0], cats_multi[1])
                st.pyplot(fig)
            else:
                st.warning("Selecciona exactamente 2 variables categóricas.")
    with tabs[9]:
        st.header("Hallazgos Clave del EDA")
        st.markdown("Resumen visual de los principales insights sobre la fuga de clientes.")

        churn_counts = df["Churn"].value_counts()
        churn_pct = (churn_counts / len(df) * 100).round(1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Clientes que se fueron (Churn=Yes)", f"{churn_counts.get('Yes', 0):,}",
                    f"{churn_pct.get('Yes', 0):.1f}% del total")
        col2.metric("Clientes retenidos (Churn=No)", f"{churn_counts.get('No', 0):,}",
                    f"{churn_pct.get('No', 0):.1f}% del total")
        col3.metric("Total clientes analizados", f"{len(df):,}")
        st.markdown("---") 

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📋 Churn por Tipo de Contrato")
            ct_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ct_contract.plot(kind="bar", ax=ax, colormap="RdYlGn_r", edgecolor="white")
            ax.set_ylabel("Porcentaje (%)")
            ax.set_title("Tasa de Churn por Tipo de Contrato")
            ax.legend(title="Churn")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Los contratos mes a mes concentran la mayor tasa de fuga.")

        with col2:
            st.markdown("##### 🌐 Churn por Servicio de Internet")
            ct_internet = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            ct_internet.plot(kind="bar", ax=ax, colormap="RdYlGn_r", edgecolor="white")
            ax.set_ylabel("Porcentaje (%)")
            ax.set_title("Tasa de Churn por Tipo de Internet")
            ax.legend(title="Churn")
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Clientes con Fiber Optic tienen mayor churn que los de DSL.")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 💸 MonthlyCharges vs Churn")
            fig, ax = plt.subplots(figsize=(6, 4))
            for val, color in zip(["Yes", "No"], ["#FF4B4B", "#3776AB"]):
                subset = df[df["Churn"] == val]["MonthlyCharges"].dropna()
                ax.hist(subset, bins=30, alpha=0.6, label=f"Churn={val}", color=color)
            ax.set_xlabel("MonthlyCharges (USD)")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Distribución MonthlyCharges por Churn")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Clientes con cargos mensuales más altos tienden a fugarse más.")

        with col2:
            st.markdown("##### ⏳ Tenure vs Churn")
            fig, ax = plt.subplots(figsize=(6, 4))
            for val, color in zip(["Yes", "No"], ["#FF4B4B", "#3776AB"]):
                subset = df[df["Churn"] == val]["tenure"].dropna()
                ax.hist(subset, bins=30, alpha=0.6, label=f"Churn={val}", color=color)
            ax.set_xlabel("Tenure (meses)")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Distribución de Tenure por Churn")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("La mayoría de los clientes que se van tienen tenure muy bajo (< 12 meses).")

        st.divider()
        st.markdown("#### 🔑 Resumen de los 5 Hallazgos Principales")

        hallazgos = [
            ("📋 Contratos mensuales = mayor riesgo",
             "Los clientes con contrato Month-to-month representan más del 40% de churn. "
             "Migrar clientes a contratos anuales es una palanca clave de retención."),
            ("💸 Cargos altos correlacionan con fuga",
             "Los clientes que se van pagan en promedio ~$20 más por mes que los que permanecen. "
             "Revisar la propuesta de valor en los segmentos de mayor costo."),
            ("⏳ Los primeros 12 meses son críticos",
             "La fuga se concentra en clientes con menos de 1 año de antigüedad. "
             "Un programa de onboarding y fidelización temprana reduciría el churn."),
            ("🌐 Fiber Optic tiene el mayor churn",
             "A pesar de ser el servicio premium, Fiber Optic concentra la mayor tasa de abandono. "
             "Posibles causas: precio elevado o insatisfacción con la calidad."),
            ("🔒 Sin seguridad online → más churn",
             "Los clientes sin OnlineSecurity ni TechSupport muestran tasas de churn más altas. "
             "Ofrecer estos servicios como valor agregado podría mejorar la retención."),
        ]

        for i, (titulo, desc) in enumerate(hallazgos, 1):
            with st.expander(f"Hallazgo {i}: {titulo}"):
                st.markdown(desc)
    st.divider()
# ===========================
#  MoDULO 4 - CONCLUSIONES
# ===========================
elif menu == "🎯 Conclusiones":
    st.title("🎯 Conclusiones y Recomendaciones Estratégicas")
    st.markdown("---")
    st.markdown("""
    A partir del Análisis Exploratorio de Datos realizado sobre el dataset
    **TelcoCustomerChurn**, se derivan las siguientes conclusiones orientadas
    a la toma de decisiones empresariales:
    """)

    conclusiones = [
        {
            "num": "1",
            "titulo": "El tipo de contrato es el factor diferenciador más importante",
            "texto": (
                "Los clientes con contratos **Month-to-month** representan la mayor proporción "
                "de fuga. La empresa debería diseñar incentivos concretos (descuentos, beneficios "
                "exclusivos) para migrar a estos clientes hacia contratos anuales o bianuales, "
                "lo que reduciría el churn de forma directa y predecible."
            ),
        },
        {
            "num": "2",
            "titulo": "La retención en los primeros 12 meses es crítica",
            "texto": (
                "El análisis de la variable `tenure` revela que la mayoría de deserciones ocurren "
                "durante el primer año. Un **programa de onboarding** robusto, con seguimiento "
                "proactivo y atención personalizada en los primeros meses, podría reducir "
                "significativamente el churn temprano."
            ),
        },
        {
            "num": "3",
            "titulo": "El precio mensual es una barrera de retención",
            "texto": (
                "Los clientes con `MonthlyCharges` elevados presentan mayor riesgo de abandono. "
                "Esto sugiere que la percepción de valor no justifica el costo para ciertos "
                "segmentos. La empresa debería evaluar **planes escalonados o bundles** que "
                "mejoren la relación costo-beneficio percibida."
            ),
        },
        {
            "num": "4",
            "titulo": "Fiber Optic requiere revisión de experiencia de cliente",
            "texto": (
                "A pesar de ser el servicio de mayor valor agregado tecnológico, Fiber Optic "
                "concentra la mayor tasa de churn. Esto podría indicar problemas de calidad, "
                "soporte técnico insuficiente o falta de alineación entre expectativas y "
                "realidad del servicio. Se recomienda una **encuesta de NPS** específica "
                "para este segmento."
            ),
        },
        {
            "num": "5",
            "titulo": "Los servicios de seguridad y soporte son palancas de fidelización",
            "texto": (
                "Los clientes sin `OnlineSecurity` y sin `TechSupport` muestran tasas de "
                "abandono considerablemente más altas. Estos servicios generan dependencia "
                "positiva y percepción de protección. Incluirlos en planes base o como "
                "**prueba gratuita de 3 meses** podría aumentar la adopción y la retención."
            ),
        },
    ]

    for c in conclusiones:
        with st.expander(f"✅ Conclusión {c['num']}: {c['titulo']}"):
            st.markdown(c["texto"])

    st.divider()
    st.markdown("#### 📊 Resumen Visual de Churn por Variables Clave")

    if "df" in st.session_state:
        df = st.session_state["df"]
        variables_cat = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, var in enumerate(variables_cat):
            ct = pd.crosstab(df[var], df["Churn"], normalize="index") * 100
            if "Yes" in ct.columns:
                ct["Yes"].plot(kind="bar", ax=axes[i], color="#FF4B4B", alpha=0.85, edgecolor="white")
                axes[i].set_title(f"% Churn por {var}")
                axes[i].set_ylabel("% Churn")
                axes[i].set_xlabel("")
                axes[i].tick_params(axis="x", rotation=25)

        plt.suptitle("Tasa de Churn (%) por Variables Categóricas Clave", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Carga el dataset en el módulo **📂 Carga de Dataset** para ver los gráficos de conclusiones.")

    st.divider()
st.markdown(
    "<center><small>Especialización Python for Analytics · 2026 "
    "© Gianella Sophia Alarcón Bardales</small></center>",
    unsafe_allow_html=True,
)
