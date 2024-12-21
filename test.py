import cv2
import mediapipe as mp
import numpy as np
import joblib  # scikit-learn modeli yüklemek için

# Modeli yükleyin
model = joblib.load("/Users/rabiayarenunlu/pythonProject/sign_language_model.pkl")

# Mediapipe el izleme modelini başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Kamera açma
cap = cv2.VideoCapture(0)

# Kameranın açıldığından emin olalım
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# her kareyi işleme
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü RGB formatına çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El tespiti işlemi
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Elin her parmağının (x, y, z) koordinatlarını almak
            coordinates = []
            for landmark in landmarks.landmark:
                # Koordinatları (x, y, z) formatında kaydediyoruz
                coordinates.append([landmark.x, landmark.y, landmark.z])

            # Koordinatları numpy array'e çevir
            coordinates = np.array(coordinates)  # Shape (21, 3)

            # Koordinatları düzleştiriyoruz (Model için uygun formata getirmek)
            coordinates = coordinates.flatten()  # Shape (63,)

            # Eğer modelin beklediği giriş 126 özellikse (sol ve sağ eldeki verileri birleştirmek için)
            # İlk başta, sağ el verisi yoksa sıfırlarla tamamlıyoruz
            right_hand_coords = np.zeros(63)  # Sağ el verisi yoksa sıfırlarla dolduruyoruz

            # Eğer daha fazla el varsa, ikinci elin koordinatlarını alıp sağ el verisi olarak ekleyebilirsiniz
            if len(result.multi_hand_landmarks) > 1:  # Eğer sağ el varsa
                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    if idx == 1:  # Sağ elin koordinatları
                        right_hand_coords = []
                        for landmark in hand_landmarks.landmark:
                            right_hand_coords.append([landmark.x, landmark.y, landmark.z])
                        right_hand_coords = np.array(right_hand_coords).flatten()  # Shape (63,)

            # Sol ve sağ el koordinatlarını birleştiriyoruz (toplamda 126 özellik)
            combined_coords = np.concatenate([coordinates, right_hand_coords])  # Shape (126,)

            # Modelin beklediği formatta bir giriş
            prediction = model.predict(combined_coords.reshape(1, -1))  # (1, 126)

            # Tahmin edilen sınıf
            predicted_class = prediction[0]  # Random Forest çıktısı direkt etiket olur
            print(f'Tahmin edilen sınıf: {predicted_class}')

            # Tahmin edilen sınıfı ekranda gösterme
            cv2.putText(frame, f'Tahmin: {predicted_class}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Görüntüyü ekranda göster
    cv2.imshow('Hand Gesture Recognition', frame)
    # 'q' tuşuna basılınca çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Kamerayı serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
