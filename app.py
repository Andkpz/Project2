from flask import Flask, render_template, request, jsonify
import joblib
import preprocessing
from class_names import get_class_name  # ← Импортируем функцию
import sklearn

app = Flask(__name__)

print(f"sklearn версия: {sklearn.__version__}")

# Загрузка модели
try:
    model = joblib.load('classifier_pipeline.pkl')
    print(f"✅ Модель загружена. Классы: {model.classes_}")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prediction_name = None  # ← Название класса
    text = ''
    processed_text = ''
    confidence = None
    error = None
    
    if request.method == 'POST':
        if model is None:
            error = "Модель не загружена"
        else:
            text = request.form.get('text', '')
            if text:
                try:
                    # 1. Предобработка
                    processed_text = preprocessing.preprocess_text(text)
                    
                    # 2. Предсказание (получаем числовой класс)
                    pred_raw = model.predict([processed_text])
                    prediction = int(pred_raw[0])  # ← Преобразуем в int
                    
                    # 3. Получаем название класса
                    prediction_name = get_class_name(prediction)  # ← Новое поле
                    
                    # 4. Уверенность
                    scores = model.decision_function([processed_text])[0]
                    confidence = float(max(scores))
                    
                    print(f"🎯 Класс: {prediction} → {prediction_name}")
                    import math
                    confidence = 1 / (1 + math.exp(-confidence))
                except Exception as e:
                    error = f"Ошибка: {str(e)}"
                    print(f"❌ {error}")
    
    return render_template('index.html', 
                         prediction=prediction,
                         prediction_name=prediction_name,  # ← Передаем в шаблон
                         text=text,
                         processed_text=processed_text,
                         confidence=confidence,
                         error=error)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint для JSON запросов"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    processed_text = preprocessing.preprocess_text(text)
    pred_raw = model.predict([processed_text])[0]
    prediction = int(pred_raw)
    scores = model.decision_function([processed_text])[0]
    
    return jsonify({
        'text': text,
        'class_id': prediction,
        'class_name': get_class_name(prediction),  # ← Название класса
        'confidence': float(max(scores))
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)