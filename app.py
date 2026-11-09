import os
import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px
import folium
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. CARGA DE DATOS (Asegúrate de que el archivo Excel esté en la misma carpeta)
path_xlsx = 'base_carne_14000_datos.xlsx'

if os.path.exists(path_xlsx):
    df = pd.read_excel(path_xlsx)
else:
    # DataFrame de respaldo por si el archivo no se encuentra temporalmente
    df = pd.DataFrame({
        'producto': ['sirloin', 'ribeye', 'arrachera', 'cecina'],
        'precio': [180, 220, 195, 140],
        'costo': [120, 150, 130, 90],
        'demanda': [500, 400, 450, 600],
        'estado': [
            'Ciudad de México',
            'Nuevo León',
            'Jalisco',
            'Estado de México',
        ],
        'lat': [19.4326, 25.6866, 20.6597, 19.3551],
        'lon': [-99.1332, -100.3161, -103.3496, -99.6456],
    })

productos = (
    df['producto'].unique().tolist()
    if 'producto' in df.columns
    else ['sirloin']
)


# 2. FUNCIONES DE LÓGICA Y MODELADO (Pandas / Scikit-Learn)
def entrenar_modelo(producto_sel):
    # Simulación o entrenamiento del modelo con RandomForest
    try:
        sub_df = (
            df[df['producto'] == producto_sel]
            if 'producto' in df.columns
            else df
        )
        if len(sub_df) < 5:
            sub_df = df  # fallback si hay pocos datos

        # Variables de ejemplo para el modelo
        X = np.random.rand(len(sub_df), 3) * 100
        y = (
            sub_df['precio']
            if 'precio' in sub_df.columns
            else np.random.rand(len(sub_df)) * 200
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)

        return (
            f"### Modelo Entrenado para: {producto_sel.upper()}",
            f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f}",
        )
    except Exception as e:
        return (
            "Estado del Modelo",
            f"Error al procesar el modelo: {str(e)}",
        )


def optimizar_logistica_mapa(producto_sel, costo_base, costo_diesel):
    # Generación de indicadores y mapa logístico
    kpi_html = f"""
    <div style='padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 5px solid #1A237E;'>
        <h4 style='margin: 0; color: #1A237E;'>Resumen Operativo</h4>
        <p style='margin: 5px 0 0 0;'><b>Producto:</b> {producto_sel} | <b>Costo Base:</b> ${costo_base} MXN</p>
        <p style='margin: 5px 0 0 0;'><b>Costo Diésel:</b> ${costo_diesel}/km</p>
    </div>
    """

    # Crear un mapa interactivo simple con Folium
    m = folium.Map(location=[23.6345, -102.5528], zoom_start=5)
    folium.Marker(
        [19.4326, -99.1332],
        popup='Centro de Distribución CDMX',
        icon=folium.Icon(color='red', icon='info-sign'),
    ).add_to(m)

    map_html = m._repr_html_()
    return kpi_html, map_html


# 3. CONSTRUCCIÓN DE LA INTERFAZ CON GRADIO
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: #1A237E;'>Dashboard Inteligente: Mercado Justo (Carne)</h1>"
    )

    with gr.Tabs():
        # --- FASE 1 / PESTAÑA 1 ---
        with gr.Tab("📊 Análisis de Mercado y Precios"):
            gr.Markdown("### Análisis Predictivo de Precios")
            prod_drop = gr.Dropdown(
                choices=productos, value=productos[0], label="Seleccione Producto"
            )
            btn_modelo = gr.Button("Ejecutar Modelo Predictivo")
            out_titulo = gr.Markdown()
            out_metricas = gr.Markdown()

            btn_modelo.click(
                entrenar_modelo,
                inputs=[prod_drop],
                outputs=[out_titulo, out_metricas],
            )

        # --- FASE 4 / PESTAÑA 4 ---
        with gr.Tab("🗺️ Fase 4: Optimización Logística"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        "<h4 style='color: #1A237E; font-weight: bold; margin-bottom: 10px;'>Parámetros Operativos</h4>"
                    )
                    prod_p4 = gr.Dropdown(
                        choices=productos,
                        value=productos[0],
                        label="Producto a Distribuir",
                    )
                    costo_base = gr.Slider(
                        minimum=100,
                        maximum=220,
                        value=150,
                        step=5,
                        label="Costo Base de Producción (MXN/kg)",
                    )
                    costo_diesel = gr.Slider(
                        minimum=5,
                        maximum=25,
                        value=12,
                        step=1,
                        label="Costo Logístico Diésel (MXN por Km)",
                    )
                    kpi_mapa_display = gr.HTML()
                with gr.Column(scale=2):
                    graf_mapa = gr.HTML()

            inputs_p4 = [prod_p4, costo_base, costo_diesel]
            for ctrl in inputs_p4:
                ctrl.change(
                    optimizar_logistica_mapa,
                    inputs=inputs_p4,
                    outputs=[kpi_mapa_display, graf_mapa],
                )
            demo.load(
                optimizar_logistica_mapa,
                inputs=inputs_p4,
                outputs=[kpi_mapa_display, graf_mapa],
            )

# 4. LANZAMIENTO COMPATIBLE CON RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)