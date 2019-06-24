import numpy as np
import cv2

class videoStream():
  def __init__(self, filepath, DNN):
    self.video_stream = cv2.VideoCapture(filepath)
    self.frame_width = self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.frame_height = self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.users = {}
    self.face_rects = []
    self.fps = self.video_stream.get(cv2.CAP_PROP_FPS)

    if DNN == "CAFFE":
        modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "deploy.prototxt"
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    self.open_video_stream()

  def open_video_stream(self):
    frame_count = 0
    while True:
      ret, self.frame = self.video_stream.read()
      if self.frame is None:
        break

      self.age_users_dict()

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
    conf_threshold = 0.8
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    self.net.setInput(blob)
    detections = self.net.forward()
    face_rects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * self.frame_width)
            y1 = int(detections[0, 0, i, 4] * self.frame_height)
            x2 = int(detections[0, 0, i, 5] * self.frame_width)
            y2 = int(detections[0, 0, i, 6] * self.frame_height)
            face_rects.append((x1, y1, x2-x1, y2-y1))
    return face_rects

  def draw_face_rects(self, frame, faces):
    for (x, y, w, h) in faces:
      small_bot = (int(0.45*w)+x, int(0.05*h)+y)
      small_top = (x+int(0.55*w), y+int(0.15*h))
      self.update_users((x, y, w, h), small_bot, small_top)
      cv2.rectangle(frame, small_bot, small_top, (0, 255, 0), 2)
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
    rect_centre = (int((2*face_rect[0]+face_rect[2])/2), int((2*face_rect[1]+face_rect[3])/2))
    cropped_rect = self.frame[rect_bot[1]:rect_top[1], rect_bot[0]:rect_top[0]]
    cv2.circle(self.frame, rect_centre,1, (0, 0, 255), 2)
    cv2.rectangle(self.frame, (face_rect[0],face_rect[1]), (face_rect[0]+face_rect[2], face_rect[1]+face_rect[3]), (0, 0, 255), 2)
    for key, values in self.users.copy().items():
      if (values['frame'][0] < rect_centre[0] and rect_centre[0] < values['frame'][0]+values['frame'][2]) and (values['frame'][1] <rect_centre[1] and rect_centre[1] < values['frame'][1] + values['frame'][3]):
        self.draw_text(self.frame, str(key), rect_centre, 100)
        self.users[key]['frame'] = face_rect
        self.users[key]['colour_hstry'] = self.update_colour_hstry(self.users[key]['colour_hstry'], cropped_rect.mean())
        self.users[key]['frames_since_update'] = 0
        return 0

    max_key = max(list(self.users.keys()))+1 if (len(self.users) != 0) else 0
    self.users[max_key] = {'frame': face_rect, 'colour_hstry': [0 for _ in range(int(self.fps*3))], 'frames_since_update': 0, 'filtered_signal': [0 for _ in range(int(self.fps*3))]}

  def age_users_dict(self):
    aged_users = []
    for key, values in self.users.copy().items():
      self.users[key]['frames_since_update'] += 1
      if self.users[key]['frames_since_update'] >= int(self.fps/2):
        aged_users.append(key)
    for key in aged_users:
      del self.users[key]

  def update_colour_hstry(self, history, mean):
    if history[int(self.fps*3)-1] == 0:
      index = history.index(0)
      history.remove(0)
      history.insert(index, mean)
    else:
      history.pop(0)
      history.append(mean)
    return history

  def filter_user_signals(self):
    for user in self.users.keys():
      raw_signal = self.users[user]['colour_hstry']
      if raw_signal[-1] != 0:
        self.users[user]['filtered_signal'] = filter_signal(raw_signal, self.fps)

def apply_bandpass(spectrum, sample_rate, high, low):
    n = spectrum.size
    high = high/sample_rate * n/2
    low = low/sample_rate * n/2
    filtered_signal = signal.copy()
    filtered_spectrum = [spectrum[i] if i >= low and i <= high else 0.0 for i in range(n)]
    return filtered_spectrum

def filter_signal(signal, sample_rate):
    coefs = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1./sample_rate)

    filtered_spectrum = apply_bandpass(coefs, sample_rate, high=1.2, low=0.8)
    filtered_signal = np.irfft(filtered_spectrum, signal.size)

if __name__ == '__main__':
  stream = videoStream(0, 'TF')
