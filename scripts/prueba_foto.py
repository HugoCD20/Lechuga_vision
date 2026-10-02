from ultralytics import YOLO

model = YOLO("/home/hugo/Documentos/programacion/Entrenamiento_lechugas/runs/detect/lechuga_yolo11n_150ep/weights/last.pt")

results = model("/home/hugo/Documentos/programacion/Entrenamiento_lechugas/images/Lechugas/IMG20250407113725.jpg")

results[0].show()