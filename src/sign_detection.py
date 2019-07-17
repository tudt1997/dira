import cv2
import numpy as np
from sign_classi import predict


def detect_sign(image_np):
    img = image_np[:, :]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(hsv, (0, 100, 100), (20, 255, 255))
    upper_red = cv2.inRange(hsv, (150, 100, 100), (179, 255, 255))
    red = cv2.bitwise_or(lower_red, upper_red)

    blue = cv2.inRange(hsv, (90, 100, 100), (110, 255, 255))
    # combined = cv2.bitwise_or(red, blue)
    combined = blue
    # combined = cv2.GaussianBlur(combined, (5, 5), 0)
    combined = cv2.blur(combined, (3, 3))
    # rev = cv2.bitwise_not(combined)
    # cv2.imshow("Thresholding", rev)

    cntr_frame, contours, hierarchy = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    sign_x = sign_y = sign_w = sign_h =  sign_size = 0
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if (w > 10 and h > 10 and 0.8 < h / w < 1.0 / 0.8):

            pred = predict(img[y:y + h, x:x + w])

            if pred != 0:
                # print(pred)
                sign_x = x
                sign_y = y
                sign_w = w
                sign_h = h
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                if pred == 1:
                    cv2.putText(img, 'Turn right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    sign_size = w
                else:
                    cv2.putText(img, 'Turn left', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    sign_size = -w
                break
            # else:
            #     cv2.putText(img, 'Not', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return img, (sign_x, sign_y, sign_w, sign_h, sign_size)
