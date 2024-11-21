from flask import Flask, render_template
import threading
import cv2
import mediapipe as mp
import pyautogui
import time

# Configurar Flask
app = Flask(__name__)

# Variável para controlar o rastreamento
tracking = False

# Inicializar a câmera e os modelos apenas uma vez para eficiência
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Função para rodar o rastreamento ocular
def eye_tracking():
    global tracking, cam, face_mesh
    frame_h, frame_w = None, None

    while tracking:
        _, frame = cam.read()
        if frame_h is None or frame_w is None:
            frame_h, frame_w, _ = frame.shape

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        if landmark_points:
            landmarks = landmark_points[0].landmark
            eye_landmarks = [landmarks[473], landmarks[473]]
            eye_x = int((eye_landmarks[0].x + eye_landmarks[1].x) * frame_w / 2)
            eye_y = int((eye_landmarks[0].y + eye_landmarks[1].y) * frame_h / 2)
            screen_x = screen_w * (eye_x / frame_w)
            screen_y = screen_h * (eye_y / frame_h)
            pyautogui.moveTo(screen_x, screen_y)

            left = [landmarks[145], landmarks[159]]
            if (left[0].y - left[1].y) < 0.0015:
                pyautogui.click()
                time.sleep(0.5)

        cv2.imshow('Eye Controlled Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tracking = False
            break

    cam.release()
    cv2.destroyAllWindows()

# Rota para servir o HTML
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('indexHome.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

# Rota para iniciar o rastreamento
@app.route('/start-tracking')
def start_tracking():
    global tracking
    if not tracking:
        tracking = True
        threading.Thread(target=eye_tracking).start()
    return "Rastreamento iniciado."

# Rota para parar o rastreamentos
@app.route('/stop-tracking')
def stop_tracking():
    global tracking
    tracking = False
    return "Rastreamento parado."

# Rodar a aplicação Flask
if __name__ == '__main__':
    app.run(port=5001)
