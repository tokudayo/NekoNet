import torch, cv2, time, os
import numpy as np
from models.detectors import Yolov5Detector
import torchvision.transforms as T
from sklearn.neighbors import KNeighborsClassifier as KNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

detector = Yolov5Detector()
model = torch.load('./models/effnetv2b0_128.pt', map_location=device)
model.eval()


transform = T.Compose([T.Resize((224, 224)),
                    lambda x : x/255.0,
                    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

knn = KNN(n_neighbors=1)

def read_db():
    X = []
    y = []
    for file in os.listdir('./db'):
        emb = torch.load('./db/' + file).cpu().detach().numpy()[0]
        X.append(emb)
        y.append(file)
    return X, y

X, y = read_db()
knn.fit(X, y)

stream = './data/cmm.mp4'

# Input stream setup
cap = cv2.VideoCapture(stream)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

FPS_LIMIT = 30
startTime = time.time()
# Frame cap
while(True):
    nowTime = time.time()
    if (nowTime - startTime) < 1/FPS_LIMIT:
        continue
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (ret == False):
        print("Cant capture any frame")
        continue
    # Detection (takes largest face)
    out = detector.detect(frame)
    for det in out.int():
        x1 = det[0].item()
        x2 = det[2].item()
        y1 = det[1].item()
        y2 = det[3].item()
        face = frame[y1:y2, x1:x2, :]
        face = np.ascontiguousarray(face[:, :, ::-1].transpose(2, 0, 1)) # to 3xHxW, BGR to RGB
        facetensor = torch.tensor(face, device=device)
        facetensor = transform(facetensor)
        topleft = (x1, y1)
        botright = (x2, y2)
        cv2.rectangle(frame, topleft, botright, (0, 0, 255), 4)
        emb = model(torch.unsqueeze(facetensor, 0))
        label = knn.predict(emb.cpu().detach().numpy())
        cv2.putText(frame, label[0], (topleft[0], topleft[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    startTime = time.time() # Reset time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()