import numpy as np
import cv2

class videoStream():
  def __init__(self, filepath):
    self.video_stream = cv2.VideoCapture(filepath)
    self.cascPath = "haarcascade_frontalface_default.xml"
    self.face_cascade = cv2.CascadeClassifier(self.cascPath)
    self.face_rects = []
    self.fps = self.video_stream.get(cv2.CAP_PROP_FPS)
    self.open_video_stream()

  def open_video_stream(self):
    frame_count = 0
    while True:
      ret, self.frame = self.video_stream.read()

      self.face_rects = self.detect_faces(self.frame)

      if len(self.face_rects) > 0:
        cropped_faces = self.crop_faces(self.frame, self.face_rects)
        self.frame = self.draw_face_rects(self.frame, self.face_rects)
        self.analyse_faces(self.frame, cropped_faces)
      self.draw_text(self.frame, str(frame_count), (100,100), 100)
      cv2.imshow('frame', self.frame)


      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      frame_count += 1

      if frame_count == 3*self.fps:
        frame_count = 0

    self.video_stream.release()
    cv2.destroyAllWindows()

  def detect_faces(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = self.face_cascade.detectMultiScale(
      image = gray,
      scaleFactor = 1.1,
      minNeighbors = 4,
      minSize = (30,30)
    )
    return face_rects

  def draw_face_rects(self, frame, faces):
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (int(0.25*w)+x, int(0.05*h)+y), (x+int(0.75*w), y+int(0.35*h)), (0, 255, 0), 2)
    return frame

  def crop_faces(self, frame, faces):
    cropped_faces = []
    for face in faces:
      (x, y, w, h) = face
      cropped_faces.append(frame[y:y+h, x:x+w])
    return cropped_faces

  def draw_text(self, frame, text, coords, size):
    cv2.putText(frame, text, 
      coords, 
      cv2.FONT_HERSHEY_SIMPLEX, 
      size/200,
      (255,255,255),
      2
    )

  def analyse_faces(self, frame, faces):
    i = 0
    for face in faces:
      face_rect = self.face_rects[i]
      self.draw_text(frame, str(face.mean()), (face_rect[0],face_rect[1]), face_rect[2])
      i += 1
      
if __name__ == '__main__':
  stream = videoStream(0)