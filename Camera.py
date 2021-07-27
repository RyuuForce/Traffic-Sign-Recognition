import numpy as np
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model("Straßenschilderkennung")

def Schildnamen(Klassennummer):
    if Klassennummer == 0: return 'Durchfahrt verboten'
    elif Klassennummer == 1: return 'Geschwindigkeitsbegrenzung 30KMH'
    elif Klassennummer == 2: return 'Stoppschild'
    elif Klassennummer == 3: return 'Unebene Fahrbahn'
    elif Klassennummer == 4: return 'Verbot der Einfahrt'
    elif Klassennummer == 5: return 'Vorfahrt an der naechsten Kreuzung'
    elif Klassennummer == 6: return 'Vorfahrt gewaehren'
    elif Klassennummer == 7: return 'Vorfahrtstrasse'

cap = cv2.VideoCapture(0)
bewertungen = []
i = 0

while(True):
    ret, frame = cap.read()
    frameimg = cv2.resize(frame, (32, 32))
    frameimg = cv2.cvtColor(frameimg, cv2.COLOR_BGR2GRAY)
    frame_array = np.array(frameimg)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = frame_array.reshape(1, 32, 32, 1)
    normalized_image_array = (frame_array.astype(np.float32) / 127.0) - 1
    vorhersage = model.predict(frame_array)
    vorhersageklasse = np.argmax(vorhersage, axis=-1)
    bewertung = np.amax(vorhersage)
    bewertung2 = vorhersage[0]


    cv2.imshow("cam", frame)
    print(
        "This image most likely belongs to " +
        str(Schildnamen(vorhersageklasse)) +
        " with a {:.2f} percent confidence."
        .format(100 * np.amax(bewertung))
    )
    #print(vorhersage)
    #print(vorhersageklasse)
    print(bewertung2[1])

    bewertungen.append(bewertung2[1])
    print("Min: " + str(np.amin(bewertungen) * 100))
    print("Max: " + str(np.amax(bewertungen) * 100))
    print("Durchschnitt: " + str(np.average(bewertungen) * 100))

    i += 1
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()