import os
import cv2
import mediapipe as mp
import csv

# MediaPipe Hands modülünü başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Veri seti dizini
DATA_DIR = '/Users/rabiayarenunlu/signLanguage/tr_signLanguage_dataset'

with open('hand_landmarks_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['letter', 'left_hand_landmark_0_x', 'left_hand_landmark_0_y', 'left_hand_landmark_0_z',
                  'left_hand_landmark_1_x', 'left_hand_landmark_1_y', 'left_hand_landmark_1_z',
                  'left_hand_landmark_2_x', 'left_hand_landmark_2_y', 'left_hand_landmark_2_z',
                  'left_hand_landmark_3_x', 'left_hand_landmark_3_y', 'left_hand_landmark_3_z',
                  'left_hand_landmark_4_x', 'left_hand_landmark_4_y', 'left_hand_landmark_4_z',
                  'left_hand_landmark_5_x', 'left_hand_landmark_5_y', 'left_hand_landmark_5_z',
                  'left_hand_landmark_6_x', 'left_hand_landmark_6_y', 'left_hand_landmark_6_z',
                  'left_hand_landmark_7_x', 'left_hand_landmark_7_y', 'left_hand_landmark_7_z',
                  'left_hand_landmark_8_x', 'left_hand_landmark_8_y', 'left_hand_landmark_8_z',
                  'left_hand_landmark_9_x', 'left_hand_landmark_9_y', 'left_hand_landmark_9_z',
                  'left_hand_landmark_10_x', 'left_hand_landmark_10_y', 'left_hand_landmark_10_z',
                  'left_hand_landmark_11_x', 'left_hand_landmark_11_y', 'left_hand_landmark_11_z',
                  'left_hand_landmark_12_x', 'left_hand_landmark_12_y', 'left_hand_landmark_12_z',
                  'left_hand_landmark_13_x', 'left_hand_landmark_13_y', 'left_hand_landmark_13_z',
                  'left_hand_landmark_14_x', 'left_hand_landmark_14_y', 'left_hand_landmark_14_z',
                  'left_hand_landmark_15_x', 'left_hand_landmark_15_y', 'left_hand_landmark_15_z',
                  'left_hand_landmark_16_x', 'left_hand_landmark_16_y', 'left_hand_landmark_16_z',
                  'left_hand_landmark_17_x', 'left_hand_landmark_17_y', 'left_hand_landmark_17_z',
                  'left_hand_landmark_18_x', 'left_hand_landmark_18_y', 'left_hand_landmark_18_z',
                  'left_hand_landmark_19_x', 'left_hand_landmark_19_y', 'left_hand_landmark_19_z',
                  'left_hand_landmark_20_x', 'left_hand_landmark_20_y', 'left_hand_landmark_20_z',
                  'right_hand_landmark_0_x', 'right_hand_landmark_0_y', 'right_hand_landmark_0_z',
                  'right_hand_landmark_1_x', 'right_hand_landmark_1_y', 'right_hand_landmark_1_z',
                  'right_hand_landmark_2_x', 'right_hand_landmark_2_y', 'right_hand_landmark_2_z',
                  'right_hand_landmark_3_x', 'right_hand_landmark_3_y', 'right_hand_landmark_3_z',
                  'right_hand_landmark_4_x', 'right_hand_landmark_4_y', 'right_hand_landmark_4_z',
                  'right_hand_landmark_5_x', 'right_hand_landmark_5_y', 'right_hand_landmark_5_z',
                  'right_hand_landmark_6_x', 'right_hand_landmark_6_y', 'right_hand_landmark_6_z',
                  'right_hand_landmark_7_x', 'right_hand_landmark_7_y', 'right_hand_landmark_7_z',
                  'right_hand_landmark_8_x', 'right_hand_landmark_8_y', 'right_hand_landmark_8_z',
                  'right_hand_landmark_9_x', 'right_hand_landmark_9_y', 'right_hand_landmark_9_z',
                  'right_hand_landmark_10_x', 'right_hand_landmark_10_y', 'right_hand_landmark_10_z',
                  'right_hand_landmark_11_x', 'right_hand_landmark_11_y', 'right_hand_landmark_11_z',
                  'right_hand_landmark_12_x', 'right_hand_landmark_12_y', 'right_hand_landmark_12_z',
                  'right_hand_landmark_13_x', 'right_hand_landmark_13_y', 'right_hand_landmark_13_z',
                  'right_hand_landmark_14_x', 'right_hand_landmark_14_y', 'right_hand_landmark_14_z',
                  'right_hand_landmark_15_x', 'right_hand_landmark_15_y', 'right_hand_landmark_15_z',
                  'right_hand_landmark_16_x', 'right_hand_landmark_16_y', 'right_hand_landmark_16_z',
                  'right_hand_landmark_17_x', 'right_hand_landmark_17_y', 'right_hand_landmark_17_z',
                  'right_hand_landmark_18_x', 'right_hand_landmark_18_y', 'right_hand_landmark_18_z',
                  'right_hand_landmark_19_x', 'right_hand_landmark_19_y', 'right_hand_landmark_19_z',
                  'right_hand_landmark_20_x', 'right_hand_landmark_20_y', 'right_hand_landmark_20_z']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Başlıkları yaz
    writer.writeheader()

    # Veri seti dizinini tara
    for split in ['train', 'test']:  # train ve test klasörlerini işleme
        split_dir = os.path.join(DATA_DIR, split)

        for label_dir in os.listdir(split_dir):  # A, B, C harfleri gibi
            if label_dir == '.DS_Store':  # Gizli dosyaları yok say
                continue

            label_dir_path = os.path.join(split_dir, label_dir)
            # Harf etiketi, dizin adından alınır
            label = label_dir

            # Bu etiketle ilgili tüm resimleri işlemeye başla
            for img_path in os.listdir(label_dir_path):
                if img_path == '.DS_Store':  # Gizli dosyaları yok say
                    continue

                img_path_full = os.path.join(label_dir_path, img_path)

                # Resmi oku ve kontrol et
                img = cv2.imread(img_path_full)
                if img is None:
                    print(f"Resim yüklenemedi: {img_path_full}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Mediapipe ile el tespiti
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    # Sol ve sağ el koordinatlarını çıkar
                    row_data = {'letter': label}
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_label = 'left' if hand_idx == 0 else 'right'

                        for i, landmark in enumerate(hand_landmarks.landmark):
                            row_data[f'{hand_label}_hand_landmark_{i}_x'] = landmark.x
                            row_data[f'{hand_label}_hand_landmark_{i}_y'] = landmark.y
                            row_data[f'{hand_label}_hand_landmark_{i}_z'] = landmark.z

                    writer.writerow(row_data)

print("Veri işleme tamamlandı.")