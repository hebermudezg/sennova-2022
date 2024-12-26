from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.graph_objects as go
import pickle

app = Flask(__name__)

# Cargar el modelo de machine learning entrenado
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# Define las columnas esperadas
expected_columns = [
    'diasactividadfisicamoderada', 'minutosactividadfisicamoderada', 'porcionesfrutaspordia', 
    'porcionesverduraspordia', 'edad', 'imc', 'diabetes_No', 'diabetes_Sí', 'alcohol_No', 
    'alcohol_Sí', 'cancer_No', 'cancer_Sí', 'insuficienciacardiaca_No', 'insuficienciacardiaca_Sí', 
    'epoc_No', 'epoc_Sí', 'estadocivil_Casado', 'estadocivil_Divorciado', 'estadocivil_Separado', 
    'estadocivil_Soltero', 'estadocivil_Unión Libre', 'estadocivil_Viudo'
]

@app.route('/')
def formulario():
    return render_template('form.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    # Extraer y convertir los datos del formulario
    data = {
        'diasactividadfisicamoderada': int(request.form.get('dias', 5)),
        'minutosactividadfisicamoderada': int(request.form.get('minutos', 120)),
        'porcionesfrutaspordia': int(request.form.get('porciones_fruta', 2)),
        'porcionesverduraspordia': int(request.form.get('porciones_verdura', 3)),
        'edad': int(request.form.get('edad', 64)),
        'imc': float(request.form.get('imc', 24.9)),
        'diabetes': 1 if request.form.get('diabetes', 'No') == 'Sí' else 0,
        'alcohol': 1 if request.form.get('alcohol', 'No') == 'Sí' else 0,
        'cancer': 1 if request.form.get('cancer', 'No') == 'Sí' else 0,
        'insuficienciacardiaca': 1 if request.form.get('insuficienciacardiaca', 'No') == 'Sí' else 0,
        'epoc': 1 if request.form.get('epoc', 'No') == 'Sí' else 0,
        'estadocivil': request.form.get('estadocivil', 'Soltero')
    }

    # Convertir a DataFrame y obtener dummies
    nuevos_df = pd.DataFrame([data])
    nuevos_df = pd.get_dummies(nuevos_df, columns=['diabetes', 'alcohol', 'cancer', 'insuficienciacardiaca', 'epoc', 'estadocivil'])

    # Alinear las columnas de `nuevos_df` con `expected_columns` y rellenar con 0
    nuevos_df = nuevos_df.reindex(columns=expected_columns, fill_value=0)

    try:
        # Realizar la predicción
        y_proba = model.predict_proba(nuevos_df)[0][1]
        probabilidad_riesgo = round(y_proba * 100, 2)  # Convertir a porcentaje
        resultado_html = f"<b>{probabilidad_riesgo}% de riesgo de caída</b>"
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({'resultado_html': "Error en la predicción", 'grafico_html': ""})

    # Genera el gráfico de riesgo con Plotly en porcentaje
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad_riesgo,  # Usar el valor en porcentaje
        title={'text': "Nivel de riesgo estimado"},
        gauge={
            'axis': {'range': [0, 100]},  # Escala de 0 a 100
            'steps': [
                {'range': [0, 40], 'color': "whitesmoke"},
                {'range': [40, 70], 'color': "gold"},
                {'range': [70, 100], 'color': "tomato"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': probabilidad_riesgo}
        }
    ))

    # Convertir el gráfico a HTML
    grafico_html = fig.to_html(full_html=False)

    return jsonify({'resultado_html': resultado_html, 'grafico_html': grafico_html})

# Punto de entrada principal para producción
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
