import cv2
import numpy as np
import json as json
import keras
import time


model = keras.models.load_model('models/CK48dataset.h5')

try:
    face_haar_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade classifiers")

# Create a list to store predictions for each frame
predictions_list = []

def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    start_time = time.time()

    while (time.time() - start_time) < 45 :
        try:
            # Capture frame by frame
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = np.array(frame)
                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 4)

                for (x, y, w, h) in faces_detected:
                    frame = np.array(frame)
                    roi_gray = gray_img[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    roi_color = cv2.resize(roi_color, (80, 80))
                    facess = face_haar_cascade.detectMultiScale(roi_gray)
                    if len(facess) == 0:
                        print("Face not detected")
                    else:
                        for (ex, ey, ew, eh) in facess:
                            face_roi = roi_color[ey: ey + eh, ex:ex + ew]
                            final_image = cv2.resize(face_roi, (80, 80))
                            final_image = np.expand_dims(final_image, axis=0)
                            final_image = final_image / 255.0

                        # Get predictions for the current frame and append to the list
                        predictions = model.predict(final_image)
                        predictions_list.append(predictions)

                        max_index = np.argmax(predictions[0])
                        highest_prediction_value = predictions.max(1) * 100.0

                        emotions = ['angry', 'fear', 'happy', 'sad']
                        predicted_emotion = emotions[max_index]
                        display_percentage_of_emotion = str(predicted_emotion) + ": " + str(highest_prediction_value)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

        except Exception as e:
            print(str(e))
    camera.release()
    cv2.destroyAllWindows()

# After processing all the frames, calculate the average predictions
average_predictions = np.mean(predictions_list, axis=0)
average_predictions_list = average_predictions.tolist()
average_predictions_json = json.dumps(average_predictions_list)
with open('./average_video_prediction.json', 'w') as file:
    file.write(average_predictions_json)
