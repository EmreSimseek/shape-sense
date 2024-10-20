from util import  ShapeDetector, get_red_mask, get_green_mask, create_rectangle
import  cv2

cap = cv2.VideoCapture(0)
top_left, bottom_right = create_rectangle(cap, scale=0.4, aspect_ratio=1.5)

detector = ShapeDetector()
color_name = "red"
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not retrieve frame from the camera. Check the connection.")
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR'yi HSV'ye çevir

    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 3) # Merkez dikdörtgen

    # Klavye girişlerini kontrol et
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' tuşuna basıldığında çık
        break
    elif key == ord('1'):  # '1' tuşuna basıldığında kırmızı maskeye geç
        color_name = "red"
        print("Kırmızı maske seçildi.")
    elif key == ord('2'):  # '2' tuşuna basıldığında yeşil maskeye geç
        color_name = "green"
        print("Yeşil maske seçildi.")

    if color_name == "red":
        mask = get_red_mask(hsv_image)
    elif color_name == "green":
        mask = get_green_mask(hsv_image)
    else:
        print("Geçersiz renk seçimi!")
        break

    processed_frame = detector.process_frame(mask,frame,top_left,bottom_right)

    # Görüntüleri göster
    cv2.imshow("Maskfirst", mask)
    cv2.imshow("fm", processed_frame)

# Kamera serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()