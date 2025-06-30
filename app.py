from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import logging

logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        if not data:
            return jsonify({'error': 'Data historis tidak ditemukan'}), 400

        df_historis = pd.DataFrame(data)
        required_columns = ['record_date', 'afdeling1', 'afdeling2', 'afdeling3', 'fertilizer_usage', 'rainfall', 'infection_level']
        if not all(col in df_historis.columns for col in required_columns):
            return jsonify({'error': f'Data tidak memiliki kolom yang diperlukan: {required_columns}'}), 400

        if len(df_historis) < 30:
            return jsonify({'error': 'Data historis minimal 30 baris'}), 400

        # Isi nilai kosong dengan 0 (atau bisa juga dengan rata-rata kolom)
        df_historis = df_historis.fillna(0)

        # Cek ulang, jika masih ada NaN, tolak request
        if df_historis.isnull().values.any():
            return jsonify({'error': 'Data historis mengandung nilai kosong/NaN'}), 400

        feature_cols = ['afdeling1', 'afdeling2', 'afdeling3', 'fertilizer_usage', 'rainfall', 'infection_level']
        df_features = df_historis[feature_cols]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features)

        X_train, y_train = [], []
        for i in range(30, len(scaled_data)):
            X_train.append(scaled_data[i-30:i])
            y_train.append(scaled_data[i, :3])

        X_train, y_train = np.array(X_train), np.array(y_train)

        if X_train.shape[0] == 0:
            return jsonify({'error': 'Data historis tidak cukup untuk training model'}), 400

        # Cek NaN di data training
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            return jsonify({'error': 'Data training mengandung NaN'}), 400

        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.3),
            LSTM(64, activation='relu'),
            Dropout(0.3),
            Dense(3)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluasi model
        y_pred_train = model.predict(X_train, verbose=0)
        if np.isnan(y_pred_train).any():
            return jsonify({'error': 'Hasil prediksi model mengandung NaN'}), 500

        mse = mean_squared_error(y_train, y_pred_train)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_train, y_pred_train)

        # Prediksi 30 hari ke depan
        last_sequence = scaled_data[-30:]
        future_predictions = []
        for _ in range(30):
            prediction = model.predict(np.array([last_sequence]), verbose=0)
            if np.isnan(prediction).any():
                return jsonify({'error': 'Prediksi masa depan mengandung NaN'}), 500
            future_predictions.append(prediction[0])
            prediction_with_dummy = np.zeros(last_sequence.shape[1])
            prediction_with_dummy[:3] = prediction[0]
            last_sequence = np.append(last_sequence[1:], [prediction_with_dummy], axis=0)

        future_predictions = np.array(future_predictions)
        temp_data = np.zeros((30, len(feature_cols)))
        temp_data[:, :3] = future_predictions
        future_predictions = scaler.inverse_transform(temp_data)[:, :3]

        start_date = pd.to_datetime(df_historis['record_date'].iloc[-1]) + pd.Timedelta(days=1)
        prediction_dates = pd.date_range(start=start_date, periods=30)

        predictions = []
        for i in range(30):
            predictions.append({
                'prediction_date': str(prediction_dates[i].date()),
                'afdeling1': float(future_predictions[i, 0]),
                'afdeling2': float(future_predictions[i, 1]),
                'afdeling3': float(future_predictions[i, 2])
            })

        evaluation = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'notes': (
                "Model memiliki performa sangat baik." if mse < 0.01 else
                "Model memiliki performa baik." if mse < 0.05 else
                "Model memiliki performa cukup baik." if mse < 0.1 else
                "Model perlu ditingkatkan."
            )
        }

        target_produksi = 5000
        rekomendasi = []
        afd_names = ['afdeling1', 'afdeling2', 'afdeling3']
        rekomendasi_text = [
            "Lakukan peremajaan tanaman untuk meningkatkan produktivitas.",
            "Optimalkan penggunaan pupuk dan irigasi untuk mempertahankan produktivitas.",
            "Pastikan tanaman muda mendapatkan perawatan yang optimal untuk mendukung pertumbuhan."
        ]
        for i, afd in enumerate(afd_names):
            if future_predictions[0, i] < target_produksi:
                rekomendasi.append({
                    'afdeling': afd,
                    'prediction_date': str(prediction_dates[0].date()),
                    'condition': 'Penurunan produksi',
                    'recommendation': rekomendasi_text[i]
                })

        return jsonify({
            'predictions': predictions,
            'evaluation': evaluation,
            'recommendations': rekomendasi
        }), 200

    except Exception as e:
        logging.error("Terjadi kesalahan saat prediksi", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan di server Flask'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
