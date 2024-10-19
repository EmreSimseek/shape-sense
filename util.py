import cv2
import numpy as np
import  time
class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.threshold_time = 5

    def start(self):
        self.start_time = time.time()

    def update(self):
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0

    def has_reached_threshold(self):
       return self.elapsed_time >= self.threshold_time

class ShapeDetector:
    def __init__(self, min_area=500, max_area=50000, epsilon_factor=0.02):
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon_factor = epsilon_factor
        self.timer = Timer()

    def detect_shape(self, contour):
        """Konturun düzgünlüğünü ve şeklini tespit et."""
        hull = cv2.convexHull(contour)  # Gürültüyü önlemek için dış çerçeve
        epsilon = self.epsilon_factor * cv2.arcLength(hull, True)  # Dinamik epsilon
        approx = cv2.approxPolyDP(hull, epsilon, True)  # Basitleştirilmiş kontur

        corners = len(approx)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)

        # Circularity kontrolüne öncelik verelim

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if 0.95 <= circularity <= 1.05:
                return "Circle"  # Daire ve sadeleştirilmiş konturu döndür

        # Şekil türünü belirle
        if corners == 3:
            return "Triangle"
        elif corners == 4:
            aspect_ratio = float(cv2.boundingRect(approx)[2]) / cv2.boundingRect(approx)[3]
            return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif 5 <= corners <= 6:
            return "Pentagon" if corners == 5 else "Hexagon"
        elif 7 <= corners <= 8:
            return "Polygon"
        else:
            return  "Undetected shape"

    def draw_continuous_progress_bar(self, frame, top_left, bottom_right, progress, color):
        x1, y1 = top_left
        x2, y2 = bottom_right
        width = x2 - x1
        height = y2 - y1
        perimeter = 2 * (width + height)

        # İlerleme barının uzunluğunu hesapla
        bar_length = int(perimeter * progress)

        # Yukarı kenar (x1, y1) -> (x2, y1) arasında dolum
        if bar_length <= width:
            cv2.line(frame, (x1, y1), (x1 + bar_length, y1), color, 2)
        else:
            cv2.line(frame, (x1, y1), (x2, y1), color, 2)
            bar_length -= width

            # Sağ kenar (x2, y1) -> (x2, y2) arasında dolum
            if bar_length <= height:
                cv2.line(frame, (x2, y1), (x2, y1 + bar_length), color, 2)
            else:
                cv2.line(frame, (x2, y1), (x2, y2), color, 2)
                bar_length -= height

                # Alt kenar (x2, y2) -> (x1, y2) arasında dolum
                if bar_length <= width:
                    cv2.line(frame, (x2, y2), (x2 - bar_length, y2), color, 2)
                else:
                    cv2.line(frame, (x2, y2), (x1, y2), color, 2)
                    bar_length -= width

                    # Sol kenar (x1, y2) -> (x1, y1) arasında dolum
                    cv2.line(frame, (x1, y2), (x1, y2 - bar_length), color, 2)

    def center_shape(self, contour):
        """Konturun ağırlık merkezini (centroid) hesaplar."""
        M = cv2.moments(contour)  # Kontur için momentleri hesapla
        if M["m00"] != 0:  # Momentlerin sıfır olmadığını kontrol et (bölme hatası önlemek için)
            cx = int(M["m10"] / M["m00"])  # x koordinatı
            cy = int(M["m01"] / M["m00"])  # y koordinatı
            return (cx, cy)
        return None  # Ağırlık merkezi hesaplanamazsa None döndür

    def is_center_within_rectangle(self, center, top_left, bottom_right):
        """Kontrol et: Merkez, verilen dikdörtgenin içinde mi?"""
        cx, cy = center
        x1, y1 = top_left
        x2, y2 = bottom_right

        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    def draw_shape(self, frame, shape, contour):
        """Şekil türüne göre kontur ve merkez çizimi."""
        epsilon = self.epsilon_factor * cv2.arcLength(contour, True)

        if shape == "Circle":
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (255, 0, 0), 2)
        else:
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)

        cv2.putText(frame, shape, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    def process_frame(self, mask, frame , top_left , bottom_right):
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                shape = self.detect_shape(contour)  # Şekli tespit et return olarak şekil ismi geliyor
                center = self.center_shape(contour) # şeklin ağırlık merkezi return ediliyor
                self.draw_shape(frame, shape, contour)
                if center:
                    cv2.circle(frame, center, 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)  # Küçük mavi nokta
                    # Merkez dikdörtgen içinde mi kontrol et
                    if self.is_center_within_rectangle(center, top_left, bottom_right):
                        if not self.shape_detected:  # Eğer daha önce tespit edilmediyse
                            self.timer.start()  # Zamanlayıcıyı başlat
                            self.shape_detected = True
                        self.timer.update()  # Zamanlayıcıyı güncelle

                        if self.timer.has_reached_threshold():  # 5 saniye doldu mu?
                            print("Merkez 5 saniye boyunca dikdörtgen içinde kaldı.")
                            self.perform_action()  # Burada yeni bir işlev çağırabilirsiniz
                        progress = self.timer.elapsed_time / self.timer.threshold_time
                        progress = min(progress, 1.0)
                        color = (0, 255, 0) if np.any(mask == 255) else (0, 0, 255)
                        self.draw_continuous_progress_bar(frame, top_left, bottom_right, progress, color)
                    else:
                        # Merkez dışarıdaysa sıfırla
                        self.timer.reset()
                        self.shape_detected = False  # Tespit durumunu sıfırla
                    self.draw_shape(frame, shape, contour)

                        # Zamanlayıcıyı sağ üst köşeye yazdır
        cv2.putText(frame, f"Time: {self.timer.elapsed_time:.2f}s",
            (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return  frame

    def perform_action(self):
        """Dikdörtgen içinde 5 saniye kalındığında yapılacak işlem."""
        print("Yeni işlev çağrıldı.")  # Buraya başka bir fonksiyonu çağırabilirsiniz

def get_red_mask(hsv_image):
        lower_red1 = np.array([0, 160, 120])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 160, 120])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Küçük parazitleri temizlemek için erosion ve dilation işlemi
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        red_mask = cv2.dilate(red_mask, kernel, iterations= 1)

        return red_mask

def get_green_mask(hsv_image):
        lower_green = np.array([35, 100, 50])
        upper_green = np.array([85, 255, 255])

        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.erode(green_mask, kernel, iterations=1)
        green_mask = cv2.dilate(green_mask, kernel, iterations=1)

        return green_mask

def create_rectangle(cap, scale, aspect_ratio):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center_x = width / 2
    center_y = height / 2

    rect_width = int(scale * width)
    rect_height = int(rect_width / aspect_ratio)

    top_left_x = int(center_x - (rect_width / 2))
    top_left_y = int(center_y - (rect_height / 2))
    bottom_right_x = int(center_x + (rect_width / 2))
    bottom_right_y = int(center_y + (rect_height / 2))

    return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)
