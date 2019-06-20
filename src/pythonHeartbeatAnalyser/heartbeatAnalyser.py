import numpy as np
import cv2

class videoStream():
  def __init__(self, filepath):
    self.video_stream = cv2.VideoCapture(filepath)
    self.cascPath = "haarcascade_frontalface_default.xml"
    self.users = {}
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
      scaleFactor = 1.15,
      minNeighbors = 4,
      minSize = (30,30)
    )
    return face_rects

  def draw_face_rects(self, frame, faces):
    for (x, y, w, h) in faces:
      small_bot = (int(0.45*w)+x, int(0.05*h)+y)
      small_top = (x+int(0.55*w), y+int(0.15*h))
      self.update_users((x, y, w, h), small_bot, small_top)
      cv2.rectangle(frame, small_bot, small_top, (0, 255, 0), 2)
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
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

  def update_users(self, face_rect, rect_bot, rect_top):
    rect_centre = (int((rect_bot[0]+rect_top[0])/2), int((rect_bot[1]+rect_top[1])/2))
    cropped_rect = self.frame[rect_bot[1]:rect_top[1], rect_bot[0]:rect_top[0]]
    for key, values in self.users.copy().items():
      self.draw_text(self.frame, str(values['colour_hstry']), (face_rect[0]+50,face_rect[1]+50), 100)
      if (values['frame'][0] < rect_centre[0] < values['frame'][0]+values['frame'][2]) and (values['frame'][1] <rect_centre[1] < values['frame'][1] + values['frame'][3]):
        self.users[key]['frame'] = face_rect
        self.users[key]['colour_hstry'] = self.update_colour_hstry(self.users[key]['colour_hstry'], cropped_rect.mean())
        self.users[key]['frames_since_update'] = 0
        return 0

    max_key = max(list(self.users.keys())) if (len(self.users) != 0) else 0
    self.users[max_key] = {'frame': face_rect, 'colour_hstry': [0 for _ in range(int(self.fps*3))], 'frames_since_update': 0}

  def update_colour_hstry(self, history, mean):
    if history[int(self.fps*3)-1] == 0:
      index = history.index(0)
      history.remove(0)
      history.insert(index, mean)
    else:
      history.pop(0)
      history.append(mean)
    return history

if __name__ == '__main__':
  stream = videoStream(0)