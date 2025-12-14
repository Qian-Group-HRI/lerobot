import cv2

print(cv2.VideoCapture())
# length = len(cv2.VideoCapture())

# print(f"OpenCV VideoCapture backend name length: {length}")
idx = 2
print(f"Trying camera index {idx} ...")


cap = cv2.VideoCapture(idx)
if not cap.isOpened():
    print(f"Camera {idx} cannot be opened.")
    cap.release()
    exit(1)

print(f"Camera {idx} opened successfully. Press 'q' to switch to next camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Camera {idx} frame not available.")
        break

    cv2.imshow(f"Camera {idx}", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

