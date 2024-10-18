import cv2
import numpy as np
class ShapeDetector:
    def __init__(self, min_area=500, max_area=50000, epsilon_factor=0.02):
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon_factor = epsilon_factor

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
            if 0.85 <= circularity <= 1.2:
                return "Circle"


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
        return None

    def process_frame(self, mask, frame):
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Bulunan kontur sayısı: {len(contours)}")  # Kontur sayısını yazdır

        for contour in contours:
            area = cv2.contourArea(contour)
            #(f"Kontur Alanı: {area}")  # Her konturun alanını yazdır

            if area > self.min_area:
                shape = self.detect_shape(contour)  # Şekli tespit et

                if shape:
                    # Konturu sadeleştirilmiş haliyle çiz
                    epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Konturu çiz
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                    # Şekil ismini sol üst köşeye yazdır
                    cv2.putText(frame, shape, (10, 30),  # Sol üst köşe
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)





        return frame




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
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)

    return red_mask

def get_green_mask(hsv_image):
    lower_green = np.array([35, 100, 50])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.erode(green_mask, kernel, iterations=1)
    green_mask = cv2.dilate(green_mask, kernel, iterations=2)

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
