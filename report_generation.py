import os
import requests
import csv

# Folder containing images
folder_path = "testing_images"
# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/detect/"

# CSV output
csv_filename = "inference_times.csv"

# Prepare CSV
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "time_taken"])

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(image_files)

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, filename)
        files = {'file': (filename, open(image_path, 'rb'), 'image/jpeg')}

        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            data = response.json()

            time_taken = data.get("time_taken", None)
            writer.writerow([filename, time_taken])

            print(f"{idx} out of {total} processed: {filename} (Time: {time_taken}s)")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
