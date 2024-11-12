import cv2
from ultralytics import YOLO
img_pth = "train/images/STONE-1021-_jpg.rf.4914ee3c6ba4e5e5df3d753b701dbd5d.jpg"
model = YOLO("runs/detect/train2/weights/best.pt")
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)

