import os
import csv
from tqdm import tqdm

from preprocessing import preprocess_audio
from feature_extraction import extract_features


DATASET_PATH = os.path.join("..", "dataset")
OUTPUT_FILE = "../metadata.csv"


def build_metadata():

    print("Scanning dataset...")

    rows = []
    id_counter = 1

    feature_names = []

    # 13 MFCC mean + std
    for i in range(13):
        feature_names.append(f"mfcc_{i+1}_mean")
    for i in range(13):
        feature_names.append(f"mfcc_{i+1}_std")

    # Spectral (3 x 2)
    spectral = ["centroid", "bandwidth", "rolloff"]
    for name in spectral:
        feature_names.append(f"{name}_mean")
        feature_names.append(f"{name}_std")

    # RMS
    feature_names.append("rms_mean")
    feature_names.append("rms_std")

    # Chroma (12 x 2)
    for i in range(12):
        feature_names.append(f"chroma_{i+1}_mean")
    for i in range(12):
        feature_names.append(f"chroma_{i+1}_std")

    # ZCR
    feature_names.append("zcr_mean")
    feature_names.append("zcr_std")

    header = ["id", "file_name", "instrument"] + feature_names

    # ==========================
    # DUYỆT DATASET
    # ==========================

    for instrument in os.listdir(DATASET_PATH):

        instrument_path = os.path.join(DATASET_PATH, instrument)

        if not os.path.isdir(instrument_path):
            continue

        print(f"\nProcessing class: {instrument}")

        for file in tqdm(os.listdir(instrument_path)):

            if not file.endswith(".aiff"):
                continue

            file_path = os.path.join(instrument_path, file)

            try:
                # 1️⃣ Preprocess
                y = preprocess_audio(file_path)

                # 2️⃣ Extract 60 features
                features = extract_features(y)

                # 3️⃣ Tạo row
                row = [
                    id_counter,
                    file,
                    instrument
                ] + features

                rows.append(row)
                id_counter += 1

            except Exception as e:
                print("Error processing:", file_path)
                print(e)

    # ==========================
    # GHI CSV
    # ==========================

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("\n===================================")
    print("DONE!")
    print("Total samples:", len(rows))
    print("Saved to:", OUTPUT_FILE)
    print("===================================")


if __name__ == "__main__":
    build_metadata()
