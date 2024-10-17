import cv2
import numpy as np
class ShapeDetector:
    def __init__(self, min_area=500, max_area=50000, epsilon_factor=0.02):
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon_factor = epsilon_factor

    def detect_shape(self, contour):
        """Konturun düzgünlüğünü ve şeklini tespit et."""
        hull = cv2.convexHull(contour)
        epsilon = self.epsilon_factor * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        corners = len(approx)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)

        # Alan ve çevre oranı ile filtreleme
        if perimeter == 0 or area < self.min_area or area > self.max_area:
            return None

        # Kontur boyutu kontrolü (bounding box)
        x, y, w, h = cv2.boundingRect(approx)
        if w < self.min_area or h < self.max_area:
             return None  # Çok küçük konturları atla

        # Şekil türünü belirle
        if corners == 3:
            return "Triangle"
        elif corners == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif corners == 5:
            return "Pentagon"
        elif 6 <= corners <= 8:
            return "Polygon"
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if 0.75 <= circularity <= 1.2:
                return "Circle"
        return None

    def process_frame(self, mask, frame, color_name):
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                shape = self.detect_shape(contour)  # Şekli tespit et

                if shape:  # Şekil algılandıysa kontur çiz ve adını ekrana yaz
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)



        return frame


def get_color(colorname):
    if colorname == "red":
        lower1, upper1 = lower_red()
        lower2, upper2 = upper_red()
        return (lower1, upper1), (lower2, upper2)
    elif colorname == "green":
        return color_green()
    else:
        raise ValueError("Choose 'red' or 'green' as the color name.")

def lower_red():
    lower_red1 = (0, 120, 70)  # İlk kırmızı aralık alt sınır
    upper_red1 = (10, 255, 255)  # İlk kırmızı aralık üst sınır
    return lower_red1, upper_red1

def upper_red():
    lower_red2 = (170, 120, 70)  # İkinci kırmızı aralık alt sınır
    upper_red2 = (180, 255, 255)  # İkinci kırmızı aralık üst sınır
    return lower_red2, upper_red2

def color_green():
    lower_green = (35, 100, 50)  # Yeşil için alt sınır
    upper_green = (85, 255, 255)  # Yeşil için üst sınır
    return lower_green, upper_green

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
