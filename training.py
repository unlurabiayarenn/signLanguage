import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Modeli kaydetmek için
import numpy as np

# CSV dosyasını yükle
data = pd.read_csv('hand_landmarks_data.csv')

def combine_hands_coordinates(row):
    left_hand_coords = row[1:64].values
    right_hand_coords = row[64:127].values

    # Eğer sağ el verisi yoksa, sıfırlarla dolduruyoruz (sol elde olanlar sıfırlanmış olur)
    if len(right_hand_coords) == 0:
        right_hand_coords = np.zeros(63)

    # Sol ve sağ el koordinatlarını birleştiriyoruz (toplamda 126 özellik)
    combined_coords = np.concatenate([left_hand_coords, right_hand_coords])

    return combined_coords

# Data Augmentation İşlemleri
def add_noise(coords, noise_level=0.01):
    noise = np.random.normal(0, noise_level, coords.shape)
    return coords + noise

def rotate_coords(coords, angle=10):
    radians = np.radians(angle)
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    coords_reshaped = coords.reshape(-1, 3)
    rotated_coords = coords_reshaped.copy()
    for i in range(len(coords_reshaped)):
        x, y = coords_reshaped[i][:2]  # Z ekseni sabit kalır
        rotated_coords[i][:2] = np.dot(rotation_matrix, [x, y])
    return rotated_coords.flatten()

def mirror_coords(coords):
    coords_reshaped = coords.reshape(-1, 3)
    mirrored_coords = coords_reshaped.copy()
    mirrored_coords[:, 0] = -mirrored_coords[:, 0]  # X ekseninde ayna görüntüsü
    return mirrored_coords.flatten()

def scale_coords(coords, scale_factor=1.1):
    return coords * scale_factor

# Belirli etiketler için augment işlemi
target_labels = ['Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']
augmented_data = []
augmented_labels = []

for _, row in data.iterrows():
    label = row['letter']
    if label in target_labels:
        coords = combine_hands_coordinates(row)
        # Augmentasyon işlemleri
        augmented_data.append(add_noise(coords))
        augmented_labels.append(label)
        augmented_data.append(rotate_coords(coords))
        augmented_labels.append(label)
        augmented_data.append(mirror_coords(coords))
        augmented_labels.append(label)
        augmented_data.append(scale_coords(coords))
        augmented_labels.append(label)

# Orijinal veriyi işleme
original_data = np.array(data.apply(combine_hands_coordinates, axis=1).tolist())
original_labels = data['letter'].values

# Augmented veriyi ekleme
if len(augmented_data) > 0:
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)
    X_augmented = np.vstack([original_data, augmented_data])
    y_augmented = np.hstack([original_labels, augmented_labels])
else:
    print("Hiçbir augmented veri oluşturulamadı. Sadece orijinal veri kullanılacak.")
    X_augmented = original_data
    y_augmented = original_labels

# Eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# RandomForestClassifier ile modelin eğitilmesi
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Modelin doğruluğunu test etme
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy * 100:.2f}%")

# Modeli kaydetme
joblib.dump(rf_classifier, 'sign_language_model.pkl')
print("Model kaydedildi.")
