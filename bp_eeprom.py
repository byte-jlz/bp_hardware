import cv2, numpy as np, requests

r = requests.get("http://192.168.1.8:8080/shot.jpg", timeout=5); r.raise_for_status()
frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("bp.jpg", frame)
print("saved", frame.shape)