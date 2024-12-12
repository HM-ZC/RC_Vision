import cv2
import numpy as np
image_path = "3.png"
image = cv2.imread(image_path)
orig = image.copy()
def nothing(x):
    pass
cv2.namedWindow("adjust")
cv2.createTrackbar("scale", "adjust", 82, 200, nothing)
cv2.createTrackbar("canny1", "adjust", 156, 255, nothing)
cv2.createTrackbar("canny2", "adjust", 255, 255, nothing)
cv2.createTrackbar("kernel", "adjust", 20, 20, nothing)
cv2.createTrackbar("area_min", "adjust", 1000, 300000, nothing)
cv2.createTrackbar("area_max", "adjust", 89149, 300000, nothing)
cv2.createTrackbar("H_min", "adjust", 48, 180, nothing)
cv2.createTrackbar("H_max", "adjust", 71, 180, nothing)
cv2.createTrackbar("S_min", "adjust", 36, 255, nothing)
cv2.createTrackbar("S_max", "adjust", 68, 255, nothing)
cv2.createTrackbar("V_min", "adjust", 100, 255, nothing)
cv2.createTrackbar("V_max", "adjust", 255, 255, nothing)
while True:
    scale = cv2.getTrackbarPos("scale", "adjust")
    canny1 = cv2.getTrackbarPos("canny1", "adjust")
    canny2 = cv2.getTrackbarPos("canny2", "adjust")
    kernel = cv2.getTrackbarPos("kernel", "adjust")
    if kernel % 2 == 0:
        kernel += 1
    area_min = cv2.getTrackbarPos("area_min", "adjust")
    area_max = cv2.getTrackbarPos("area_max", "adjust")
    H_min = cv2.getTrackbarPos("H_min", "adjust")
    H_max = cv2.getTrackbarPos("H_max", "adjust")
    S_min = cv2.getTrackbarPos("S_min", "adjust")
    S_max = cv2.getTrackbarPos("S_max", "adjust")
    V_min = cv2.getTrackbarPos("V_min", "adjust")
    V_max = cv2.getTrackbarPos("V_max", "adjust")
    if scale != 100:
        if scale < 1:
            scale = 1
        width = max(1, int(image.shape[1] * scale / 100))
        height = max(1, int(image.shape[0] * scale / 100))
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image.copy()
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([H_min, S_min, V_min])
    heigher_color = np.array([H_max, V_max, V_max])
    color_mask = cv2.inRange(hsv, lower_color, heigher_color)
    color_result = cv2.bitwise_and(resized_image, resized_image, mask=color_mask)
    gray = cv2.cvtColor(color_result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny1, canny2)
    Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, Kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = resized_image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_min < area < area_max:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("adjust", result)
    cv2.imshow("hsv", color_result)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
