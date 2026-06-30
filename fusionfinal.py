import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
import threading

# DEVICE AND FACE DETECTOR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# RESNET50 

NUM_CLASSES = 7
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
ema_state = torch.load("best_rafdb_resnet50.pth", map_location=DEVICE)
model.load_state_dict(ema_state, strict=False)
model = model.to(DEVICE)
model.eval()

emotion_labels = {0:"Anger",1:"Disgust",2:"Fear",3:"Happiness",4:"Sadness",5:"Surprise",6:"Neutral"}
emotion_stress_map = {0:80,1:75,2:90,3:10,4:70,5:50,6:35}


# GLOBAL RESULT

camera_stress_result = 0
stop_event = threading.Event()

# CAMERA STRESS FUNCTION
def get_camera_stress():
    global camera_stress_result
    cap = cv2.VideoCapture(0)
    stress_history = []

    while not stop_event.is_set(): 
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        frame_stress_values = []

        for (x,y,w,h) in faces:
            face = cv2.resize(frame[y:y+h,x:x+w],(224,224))
            face_tensor = torch.tensor(face).permute(2,0,1).unsqueeze(0).float()/255.0
            face_tensor = face_tensor.to(DEVICE)
            with torch.no_grad():
                outputs = model(face_tensor)
                pred_class = outputs.argmax(dim=1).item()
            frame_stress_values.append(emotion_stress_map[pred_class])

        if frame_stress_values:
            stress_history.append(np.mean(frame_stress_values))
            if len(stress_history)>30:
                stress_history.pop(0)

        cv2.putText(frame,f"Stress: {np.mean(stress_history) if stress_history else 0:.1f}/100",(10,35),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
        cv2.imshow("Camera Stress",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    camera_stress_result = np.mean(stress_history) if stress_history else 0

# SURVEY STRESS FUNCTION 
def get_survey_stress():
    # Load dataset
    data_file = pd.read_csv('Stress Indicators Dataset for Mental Health Classification.csv')
    group_high = data_file[data_file['stress_type']==0]
    group_mid  = data_file[data_file['stress_type']==1]
    group_low  = data_file[data_file['stress_type']==2]
    target_size = len(group_mid)
    balanced_data = pd.concat([group_mid,
                               group_high.sample(target_size, replace=True, random_state=42),
                               group_low.sample(target_size, replace=True, random_state=42)])

    features = [col for col in data_file.columns if col!="stress_type"]
    X = balanced_data[features].values
    y = 1.0 - (balanced_data['stress_type'].values.reshape(-1,1).astype(float)/2.0)
    dataset_medians = np.median(X,axis=0)
    dataset_std_devs = np.std(X,axis=0)+1e-8

    def scale_numbers(input_data):
        scaled = (input_data-dataset_medians)/dataset_std_devs
        return np.clip(scaled,-3.0,3.0)

    X_scaled = scale_numbers(X)
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(X_scaled))
    X_train, y_train = X_scaled[shuffled_idx], y[shuffled_idx]

    # Neural network
    class StudentStressModel:
        def __init__(self,input_size,hidden_nodes=64,l_rate=0.05):
            self.w1 = np.random.randn(input_size,hidden_nodes)*np.sqrt(2./input_size)
            self.b1 = np.zeros((1,hidden_nodes))
            self.w2 = np.random.randn(hidden_nodes,1)*np.sqrt(2./hidden_nodes)
            self.b2 = np.zeros((1,1))
            self.learning_rate=l_rate

        def sigmoid_func(self,x):
            return 1/(1+np.exp(-np.clip(x,-20,20)))

        def forward(self,x_input):
            self.z1 = x_input@self.w1+self.b1
            self.a1 = np.tanh(self.z1)
            self.z2 = self.a1@self.w2+self.b2
            self.output = self.sigmoid_func(self.z2)
            return self.output

        def train(self,x_data,y_data,epochs=2000):
            m = x_data.shape[0]
            for _ in range(epochs):
                pred = self.forward(x_data)
                error_val = pred - y_data
                dz2 = error_val*(pred*(1-pred))
                dw2 = (self.a1.T@dz2)/m
                db2 = np.mean(dz2,axis=0,keepdims=True)
                dz1 = (dz2@self.w2.T)*(1-self.a1**2)
                dw1 = (x_data.T@dz1)/m
                db1 = np.mean(dz1,axis=0,keepdims=True)
                for grad in [dw1,db1,dw2,db2]:
                    np.clip(grad,-0.5,0.5,out=grad)
                self.w1 -= self.learning_rate*dw1
                self.b1 -= self.learning_rate*db1
                self.w2 -= self.learning_rate*dw2
                self.b2 -= self.learning_rate*db2

    model_nn = StudentStressModel(input_size=len(features))
    print("Training survey model (camera recording in background)...")
    model_nn.train(X_train,y_train)
    print("Survey model trained.\n")

    # Ask interactive questions
    responses=[]
    print("--- Personal Stress Survey ---")
    for i,f_name in enumerate(features):
        while True:
            try:
                if "gender" in f_name.lower():
                    val=float(input(f"{f_name} (1=Male,0=Female): "))
                    val=np.clip(val,0,1)
                elif "age" in f_name.lower():
                    val=float(input(f"{f_name}: "))
                    val=np.clip(val,10,100)
                else:
                    val=float(input(f"{f_name} (1-5): "))
                    val=np.clip(val,1,5)
                responses.append(val)
                break
            except:
                print("Invalid input, try again.")

    test_input = scale_numbers(np.array(responses).reshape(1,-1))
    prob_result = model_nn.forward(test_input)[0][0]
    return prob_result*100


# RUN CAMERA AND SURVEY PARALLEL

camera_thread = threading.Thread(target=get_camera_stress)
camera_thread.start()

# Survey runs in main thread
survey_score = get_survey_stress()

# Signal camera to stop after survey finishes
stop_event.set()
camera_thread.join()
camera_score = camera_stress_result

# COMBINE RESULTS

final_stress = 0.25*camera_score + 0.75*survey_score
print("\n" + "="*40)
print(f"Camera Stress Score: {camera_score:.2f}/100")
print(f"Survey Stress Score: {survey_score:.2f}/100")
print(f"Overall Combined Stress Score: {final_stress:.2f}/100")

if final_stress >= 80:
    status="CRITICAL STRESS"
elif final_stress >= 60:
    status="HIGH STRESS"
elif final_stress >= 40:
    status="MODERATE STRESS"
else:
    status="LOW STRESS"

print(f"Overall Status: {status}")
print("="*40)
