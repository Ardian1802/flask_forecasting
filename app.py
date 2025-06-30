from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import logging 
import mysql.connector
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)

load_dotenv()
# Konfigurasi koneksi database
db_config = {
     'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
     'use_pure': True,  # Tambahan penting
}

@app.route('/test-db')
def test_db():
    try:
        conn = mysql.connector.connect(**db_config)
        return "✅ Koneksi ke database berhasil"
    except Exception as e:
        return f"❌ Gagal koneksi DB: {str(e)}"


@app.route('/predict', methods=['POST']) 
def predict():
    try:
        # Koneksi ke database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Ambil data historis dari database
        cursor.execute("""
            SELECT record_date, afdeling1, afdeling2, afdeling3, fertilizer_usage, rainfall, infection_level
            FROM historical_data
            ORDER BY record_date
        """)
        historis = cursor.fetchall()

        if not historis:
            return jsonify({'error': 'Data historis kosong'}), 400

        # Konversi data historis ke DataFrame
        df_historis = pd.DataFrame(historis)

        # Validasi kolom yang diperlukan
        required_columns = ['record_date', 'afdeling1', 'afdeling2', 'afdeling3', 'fertilizer_usage', 'rainfall', 'infection_level']
        if not all(col in df_historis.columns for col in required_columns):
            return jsonify({'error': f'Data tidak memiliki kolom yang diperlukan: {required_columns}'}), 400

        # Ambil hanya kolom fitur yang digunakan
        feature_cols = ['afdeling1', 'afdeling2', 'afdeling3', 'fertilizer_usage', 'rainfall', 'infection_level']
        df_features = df_historis[feature_cols]

        # Normalisasi data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features)

        # Persiapkan data untuk pelatihan
        X_train = []
        y_train = []

        # Gunakan 30 data terakhir untuk memprediksi data berikutnya
        for i in range(30, len(scaled_data)):
            X_train.append(scaled_data[i-30:i])  # 30 data sebelumnya sebagai input
            y_train.append(scaled_data[i, :3])  # 3 kolom target sebagai output

        X_train, y_train = np.array(X_train), np.array(y_train)

        # Bangun model LSTM dengan lapisan tambahan
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),  # Input layer
            LSTM(64, activation='relu', return_sequences=True),  # LSTM pertama
            Dropout(0.3),  # Dropout pertama
            LSTM(64, activation='relu', return_sequences=False),  # LSTM kedua
            Dropout(0.3),  # Dropout kedua
            Dense(3)  # Output layer dengan 3 target
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Callbacks untuk menghentikan pelatihan jika overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # Latih model
        history = model.fit(
            X_train, y_train,
            epochs=50,  # Maksimal 50 epoch
            batch_size=32,
            validation_split=0.2,  # Gunakan 20% data untuk validasi
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        # Evaluasi model menggunakan data pelatihan
        y_pred_train = model.predict(X_train, verbose=0)

        # Hitung MSE, RMSE, dan MAE
        mse = mean_squared_error(y_train, y_pred_train)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_train, y_pred_train)

        # Log nilai evaluasi
        logging.info(f"Nilai evaluasi: MSE={mse}, RMSE={rmse}, MAE={mae}")

        # Tentukan tanggal awal prediksi
        cursor.execute("SELECT MAX(prediction_date) AS last_prediction_date FROM predictions")
        last_prediction = cursor.fetchone()

        if last_prediction and last_prediction['last_prediction_date']:
            # Jika ada prediksi sebelumnya, gunakan tanggal terakhir dari tabel predictions
            start_date = pd.to_datetime(last_prediction['last_prediction_date']) + pd.Timedelta(days=1)
        else:
            # Jika tidak ada prediksi sebelumnya, gunakan tanggal terakhir dari data historis
            start_date = pd.to_datetime(df_historis['record_date'].iloc[-1]) + pd.Timedelta(days=1)

        # Tentukan tanggal prediksi (30 hari ke depan)
        prediction_dates = pd.date_range(start=start_date, periods=30)

        # Prediksi 30 hari ke depan
        future_predictions = []
        last_sequence = scaled_data[-30:]  # Ambil urutan terakhir

        for _ in range(30):
            prediction = model.predict(np.array([last_sequence]), verbose=0)
            future_predictions.append(prediction[0])

            # Tambahkan kolom dummy untuk melengkapi dimensi menjadi (6,)
            prediction_with_dummy = np.zeros(last_sequence.shape[1])  # Buat array dengan dimensi (6,)
            prediction_with_dummy[:3] = prediction[0]  # Masukkan 3 kolom target ke dalam array

            # Gabungkan array dengan dimensi yang sesuai
            last_sequence = np.append(last_sequence[1:], [prediction_with_dummy], axis=0)

        # Denormalisasi hanya untuk 3 kolom target
        future_predictions = np.array(future_predictions)
        temp_data = np.zeros((30, len(feature_cols)))
        temp_data[:, :3] = future_predictions
        future_predictions = scaler.inverse_transform(temp_data)[:, :3]

        # Hitung produktivitas berdasarkan luas lahan
        productivity = [
            (
                str(prediction_dates[i].date()),
                float(future_predictions[i, 0]),  # Prediksi Afdeling 1
                float(future_predictions[i, 1]),  # Prediksi Afdeling 2
                float(future_predictions[i, 2]),  # Prediksi Afdeling 3
                float(future_predictions[i, 0] / 1727),  # Produktivitas Afdeling 1
                float(future_predictions[i, 1] / 258),   # Produktivitas Afdeling 2
                float(future_predictions[i, 2] / 435)    # Produktivitas Afdeling 3
            )
            for i in range(30)
        ]

        # Simpan hasil prediksi dan produktivitas ke database
        cursor.executemany("""
            INSERT INTO predictions (
                prediction_date, 
                prediction_afdeling1, 
                prediction_afdeling2, 
                prediction_afdeling3, 
                productivity_afd1, 
                productivity_afd2, 
                productivity_afd3
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, productivity)
        conn.commit()

        # Gunakan tanggal awal prediksi sebagai tanggal evaluasi
        for i, prediction_date in enumerate(prediction_dates):
            try:
                # Tentukan catatan berdasarkan nilai evaluasi
                if mse < 0.01:
                    evaluation_notes = f"Model memiliki performa sangat baik pada tanggal {prediction_date.date()}."
                elif mse < 0.05:
                    evaluation_notes = f"Model memiliki performa baik pada tanggal {prediction_date.date()}."
                elif mse < 0.1:
                    evaluation_notes = f"Model memiliki performa cukup baik pada tanggal {prediction_date.date()}."
                else:
                    evaluation_notes = f"Model perlu ditingkatkan pada tanggal {prediction_date.date()}."

                logging.info(f"⏳ Menyimpan evaluasi untuk tanggal {prediction_date.date()}: MSE={mse}, RMSE={rmse}, MAE={mae}, Notes={evaluation_notes}")
                cursor.execute("""
                    INSERT INTO evaluations (evaluation_date, mse, rmse, mae, notes)
                    VALUES (%s, %s, %s, %s, %s)
                """, (prediction_date.date(), mse, rmse, mae, evaluation_notes))
                conn.commit()
                logging.info(f"✅ Evaluasi untuk tanggal {prediction_date.date()} berhasil disimpan ke database.")
            except mysql.connector.Error as err:
                logging.error(f"❌ Kesalahan saat menyimpan evaluasi untuk tanggal {prediction_date.date()}: {err}")

        # Target produksi untuk mendeteksi penurunan
        target_produksi = 5000  # Target produksi untuk semua afdeling

        # Ambil semua user_id dari tabel users
        cursor.execute("SELECT id FROM users")
        user_ids = [row['id'] for row in cursor.fetchall()]

        # Deteksi penurunan produksi dan buat notifikasi
        notifications = [] 
        recommendations = []
        for i, afdeling in enumerate(['afdeling1', 'afdeling2', 'afdeling3']):
            predicted_production = future_predictions[0, i]

            if predicted_production < target_produksi:  # Bandingkan dengan target produksi
                if afdeling == 'afdeling1':  # Tanaman tua
                    recommendation_text = "Lakukan peremajaan tanaman untuk meningkatkan produktivitas."
                elif afdeling == 'afdeling2':  # Tanaman produktif
                    recommendation_text = "Optimalkan penggunaan pupuk dan irigasi untuk mempertahankan produktivitas."
                elif afdeling == 'afdeling3':  # Tanaman muda
                    recommendation_text = "Pastikan tanaman muda mendapatkan perawatan yang optimal untuk mendukung pertumbuhan."

                # Simpan rekomendasi ke tabel recommendations
                recommendations.append((i + 1, prediction_dates[0].date(), "Penurunan produksi", recommendation_text))

                # Simpan notifikasi ke tabel notifications untuk setiap user_id
                for user_id in user_ids:
                    notifications.append((user_id, f"Produksi {afdeling} diprediksi turun menjadi {predicted_production:.2f} (target: {target_produksi}). {recommendation_text}", 'web', None, 'pending'))

        # Simpan rekomendasi ke database
        if recommendations:
            cursor.executemany("""
                INSERT INTO recommendations (afdeling_id, record_date, `condition`, recommendation_text)
                VALUES (%s, %s, %s, %s)
            """, recommendations)
            conn.commit()

        # Simpan notifikasi ke database
        if notifications:
            cursor.executemany("""
                INSERT INTO notifications (user_id, message, notification_type, sent_at, status)
                VALUES (%s, %s, %s, %s, %s)
            """, notifications)
            conn.commit()

        # Tutup koneksi database
        cursor.close()
        conn.close()

        return jsonify({'message': 'Prediksi berhasil disimpan ke database.'}), 200

    except Exception as e:
        logging.error("Terjadi kesalahan pada endpoint /predict", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan pada server. Silakan cek log untuk detailnya.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
