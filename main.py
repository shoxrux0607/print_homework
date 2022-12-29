'''
Ushbu python loyihasida biz tasvirdagi odamlarni taniydigan mashinani o'rganish(machine learning) modelini yaratmoqchimiz. Biz loyihamizda face_recognition API va OpenCV dan foydalanamiz.
Kutubxonalar
Python - 3.x
cv2 - 4.5.2
numpy - 1.20.3
face_recognition - 1.3.0

Yuqoridagi paketlarni o'rnatish uchun quyidagi buyruqdan foydalanamiz.

pip install numpy opencv-python
pip install dlib
pip install face_recognition
'''
'''Modelni o'rgatamiz yani training
Avval kerakli modullarni import qilamiz.
Face_recognition kutubxonasi yuzni tanib olish jarayonida yordam beradigan turli yordam dasturlarini o'z ichiga oladi.'''

import face_recognition as fr
import cv2 
import numpy as np
import os

'''Endi tasvirlar (shaxslar) nomlarini va ularning yuz kodlashlarini saqlaydigan 2 ta ro'yxat yaratamiz.'''

path = "./training_rasm/"
known_names = []
known_name_encodings = []
images = os.listdir(path)

'''Yuzni kodlash - bu ko'zlar orasidagi masofa, peshonaning kengligi va boshqalar kabi yuzning farqlovchi xususiyatlari o'rtasidagi muhim o'lchovlarni ifodalovchi qiymatlar vektori.
Biz training_rasm katalogimizdagi har bir tasvirni aylanib chiqamiz, rasmdagi shaxsning ismini chiqaramiz, uning yuz kodlash vektorini hisoblaymiz va ma'lumotlarni tegishli ro'yxatlarda saqlaymiz.
'''
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

'''Modelni test ma'lumotlar to'plamida sinab ko'ring
Yuqorida aytib o'tilganidek, bizning umumiy_rasm katalogimizdagi ma'lumotlar to'plamimiz undagi barcha shaxslar bilan faqat 1 ta rasmni o'z ichiga oladi.
Cv2 imread() usuli yordamida test tasvirini o'qiydi.

'''
test_image = "./umumiy_rasm/do'stlar.jpg"
image = cv2.imread(test_image)

'''Face_recognition kutubxonasi suratda aniqlangan har bir yuzning koordinatalarini
 (chap, pastki, o'ng, yuqori) joylashtirgan face_locations() deb nomlangan foydali usulni taqdim etadi.
 Ushbu joylashuv qiymatlaridan foydalanib, biz yuz kodlashlarini osongina topishimiz mumkin
'''

face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = " "
    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match = np.argmin(face_distances)
    if matches[best_match]:
        name = known_names[best_match]
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
'''Cv2 modulining imshow() usuli yordamida tasvirni ko'rsatadi.'''

cv2.imshow("Natija", image)

'''Tasvirni imwrite() usuli yordamida joriy ishchi katalogimizga saqlaydi.'''

cv2.imwrite("./Natija_rasm.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

