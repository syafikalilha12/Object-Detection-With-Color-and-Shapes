import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(True):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 60, 230)
    contour, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    jumlah = str(len(contour))
    print("jumlah object = ",jumlah)
    result_contour = cv2.drawContours(frame,contour,-1,(0,255,0),2)

    for i in range(int(jumlah)):
        area = cv2.contourArea(contour[i])
        perimeter = cv2.arcLength(contour[i], True)
        #TR = (4*np.pi*area)/(perimeter**)
        #print("nilai perbandingan kektebalan :",TR)
    cv2.imshow("result contour",result_contour)
    cv2.imshow("canny",edge)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()