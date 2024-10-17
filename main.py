from util import  ShapeDetector, get_color, create_rectangle
import  cv2
import numpy as np

cap = cv2.VideoCapture(0)
top_left, bottom_right = create_rectangle(cap, scale=0.4, aspect_ratio=1.5)

detector = ShapeDetector(min_area=500)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame from the camera. Check the connection.")
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR'yi HSV'ye çevir
    blurred = cv2.GaussianBlur(hsv_image, (5, 5), 0)

    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2) # Merkez dikdörtgen

    # Renk sınırlarını alalım (örneğin kırmızı için)
    color_name = "red"  # İsterseniz 'green' yapabilirsiniz
    color_ranges = get_color(color_name)

    # İki kırmızı aralık için maskeleri birleştirme (kırmızı örneğinde)
    if color_name == "red":
        mask1 = cv2.inRange(blurred, color_ranges[0][0], color_ranges[0][1])
        mask2 = cv2.inRange(blurred, color_ranges[1][0], color_ranges[1][1])
        mask = cv2.bitwise_or(mask1, mask2)  # İki maskeyi birleştiriyoruz
    else:
        mask = cv2.inRange(blurred, color_ranges[0], color_ranges[1])


    processed_frame = detector.process_frame(mask,frame,color_name)

    # Görüntüleri göster
    cv2.imshow("fm", processed_frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Kamera serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()