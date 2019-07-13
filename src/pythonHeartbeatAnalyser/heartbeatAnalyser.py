import numpy as np
from scipy.signal import butter, filtfilt, lfilter
import cv2

import matplotlib.pyplot as plt

class videoStream():
  def __init__(self, filepath, DNN):
    self.LOW_PASS = 1
    self.HIGH_PASS = 1.5
    self.BUTTER_ORDER = 3
    self.NOISE_THRESHOLD = 1.5
    self.LAPLACIAN_LAYERS = 4
    self.PLOT_USER = True
    self.WINDOW_SECONDS = 3

    self.video_stream = cv2.VideoCapture(filepath)
    self.frame_width = self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.frame_height = self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.users = {}
    self.face_rects = []
    
    self.fps = self.video_stream.get(cv2.CAP_PROP_FPS)
    self.time_window = self.fps * self.WINDOW_SECONDS
    self.noise_amp = 0
    self.noise = [[0 for i in range(int(self.time_window))] for _ in range(3)]
    self.filtered_noise = [[0 for i in range(int(self.time_window))] for _ in range(3)]

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
    if self.PLOT_USER: fig, ax = self.initialise_plot()
    frame_count = 0
    while True:
      ret, self.frame = self.video_stream.read()
      if self.frame is None:
        break
      self.get_noise_amp()
      self.age_users_dict()

      self.face_rects = self.detect_faces(self.frame)
      if len(self.face_rects) > 0:
        cropped_faces = self.crop_faces(self.frame, self.face_rects)
        self.frame = self.draw_face_rects(self.frame, self.face_rects)
      self.draw_text(self.frame, str(frame_count), (100,100), 100)
      cv2.imshow('frame', self.frame)

      self.filter_user_signals()
      if self.PLOT_USER: fig, ax = self.update_plot(fig, ax)

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
      #small_bot = (int(0.35*w)+x, int(0.05*h)+y)
      #small_top = (x+int(0.65*w), y+int(0.25*h))
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

  def get_rect_centre(self, rect):
    return (int((2*rect[0]+rect[2])/2), int((2*rect[1]+rect[3])/2))

  def update_users(self, face_rect, rect_bot, rect_top):
    rect_centre = self.get_rect_centre(face_rect)
    cropped_rect = self.frame[rect_bot[1]:rect_top[1], rect_bot[0]:rect_top[0]]
    cv2.circle(self.frame, rect_centre,1, (0, 0, 255), 2)
    cv2.rectangle(self.frame, (face_rect[0],face_rect[1]), (face_rect[0]+face_rect[2], face_rect[1]+face_rect[3]), (0, 0, 255), 2)
    for key, values in self.users.copy().items():
      if (values['frame'][0] < rect_centre[0] and rect_centre[0] < values['frame'][0]+values['frame'][2]) and (values['frame'][1] <rect_centre[1] and rect_centre[1] < values['frame'][1] + values['frame'][3]):
        self.draw_text(self.frame, str(key), rect_centre, 100)
        self.draw_text(self.frame, str(values['bpm']), (face_rect[0],face_rect[1]), face_rect[2])
        (b, g, r), hsv_img = self.extract_colour_signal(cropped_rect)
        self.users[key]['frame'] = face_rect
        self.users[key]['b'] = self.update_signal(self.users[key]['b'], b)
        self.users[key]['g'] = self.update_signal(self.users[key]['g'], g)
        self.users[key]['r'] = self.update_signal(self.users[key]['r'], r)
        self.users[key]['hsv_img'] = hsv_img
        self.users[key]['frames_since_update'] = 0
        cv2.imshow(str(key), self.boost_image(hsv_img, values['concatted_signal']))

        return 0

    max_key = max(list(self.users.keys()))+1 if (len(self.users) != 0) else 0
    self.users[max_key] = {'frame': face_rect, 
        'b':[0 for _ in range(int(self.time_window))], 
        'g':[0 for _ in range(int(self.time_window))], 
        'r':[0 for _ in range(int(self.time_window))], 
        'frames_since_update': 0, 
        'filtered_signal': [[0 for _ in range(int(self.time_window))] for i in range(3)],
        'raw_signal': [0 for _ in range(int(self.time_window))], 
        'concatted_signal': [0 for _ in range(int(self.time_window))],
        'bpm': 'calculating bpm', 'hsv_img': None}

  def boost_image(self, image, filtered_signal):
     b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
     b = b + 30*filtered_signal[-1]
     g = g + 30*filtered_signal[-1]
     r = r + 30*filtered_signal[-1]
     image[:, :, 0], image[:, :, 1], image[:, :, 2] = b, g, r
     return image

  def age_users_dict(self):
    aged_users = []
    for key, values in self.users.copy().items():
      self.users[key]['frames_since_update'] += 1
      if self.users[key]['frames_since_update'] >= int(self.fps/2):
        aged_users.append(key)
    for key in aged_users:
      cv2.destroyWindow(str(key))
      del self.users[key]

  def update_signal(self, history, value):
    if history[-1] == 0:
      index = history.index(0)
      history.remove(0)
      history.insert(index, value)
    else:
      history.pop(0)
      history.append(value)
    return history

  def filter_user_signals(self):
    for key, values in self.users.copy().items():
      raw_signal = (self.users[key]['b'], self.users[key]['g'], self.users[key]['r'])
      if raw_signal[0][-1]!=0 and raw_signal[1][-1]!=0 and raw_signal[2][-1]!=0:
        face_centre = self.get_rect_centre(values['frame'])
        filtered_signal = [butter_bandpass_filter(np.array(signal_channel), self.LOW_PASS, self.HIGH_PASS, self.fps, order=self.BUTTER_ORDER) for signal_channel in raw_signal]
        self.users[key]['filtered_signal'] = filtered_signal
        self.users[key]['raw_signal'] = raw_signal
        self.users[key]['concatted_signal'] =  self.eliminate_low_freq_noise(np.sum(filtered_signal, axis=0))
        self.users[key]['bpm'], max_amp = self.extract_heartbeat(self.users[key]['concatted_signal'])

  def get_noise_amp(self):
    sample = self.frame[1:5,1:5]
    sample_signal, _ = self.extract_colour_signal(sample)
    self.noise[0] = self.update_signal(self.noise[0], sample_signal[0])
    self.noise[1] = self.update_signal(self.noise[1], sample_signal[1])
    self.noise[2] = self.update_signal(self.noise[2], sample_signal[2])
    self.filtered_noise = [butter_bandpass_filter(np.array(signal_channel), self.LOW_PASS, self.HIGH_PASS, self.fps, order=self.BUTTER_ORDER) for signal_channel in self.noise]
    noise_signal = np.sum(self.filtered_noise, axis=0)
    _, max_amp = self.extract_heartbeat(noise_signal, noise=True)
    self.noise_amp = max_amp

  def extract_heartbeat(self, signal, noise=False):
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.size, 1/self.fps)
    max_index = np.abs(np.argmax(np.abs(spectrum)))

    max_freq = np.abs(frequencies[max_index])
    max_amp = np.abs(spectrum[max_index])

    max_freq_hz = max_freq
    max_freq_bpm = max_freq_hz * 60
    return max_freq_bpm, max_amp

  def cutoff_low_amp(self, spectrum, threshold):
    threshold_indices = np.abs(spectrum) < self.NOISE_THRESHOLD*self.noise_amp
    spectrum[threshold_indices] = 0
    return spectrum

  def eliminate_low_freq_noise(self, signal):
    spectrum, frequencies = np.fft.fft(signal), np.fft.fftfreq(signal.size, 1/self.fps)
    spectrum = self.cutoff_low_amp(spectrum, self.noise_amp)
    signal = np.fft.ifft(spectrum, signal.size)
    return signal

  def extract_colour_signal(self, frame):
    img = generate_laplacian_pyramid(frame, self.LAPLACIAN_LAYERS)
    img = cv2.resize(img, (200,200))
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    return (np.mean(b), np.mean(g), np.mean(r)), img

  def initialise_plot(self):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.show()
    fig.canvas.draw()
    return fig, ax

  def update_plot(self, fig, ax):
    ax.clear()
    if 0 in self.users.keys():
      ax.plot(100*self.users[0]['concatted_signal'])
      ax.plot(np.sum(self.users[0]['raw_signal'], axis=0))
      fig.canvas.draw()
    return fig, ax


# def apply_bandpass(spectrum, sample_rate, high, low):
#     n = spectrum.size
#     high = high/sample_rate * n
#     low = low/sample_rate * n
#     filtered_signal = spectrum.copy()
#     filtered_spectrum = [spectrum[i] if i >= low and i <= high else 0.0 for i in range(n)]
#     return filtered_spectrum

# def filter_signal(signal, sample_rate):
#     coefs = np.fft.fft(signal)
#     freqs = np.fft.fftfreq(signal.size, d=1./sample_rate)

#     filtered_spectrum = apply_bandpass(coefs, sample_rate, high=2.5, low=0.8)
#     filtered_signal = np.fft.ifft(filtered_spectrum, signal.size)
#     return filtered_signal, filtered_spectrum, coefs, freqs

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, (data-np.mean(data)))
  #y = filtfilt(b, a, (data-np.mean(data)))
  return y

def generate_gaussian_pyramid(image, levels):
  g_pyr = [image]
  for i in range(levels):
    image = cv2.pyrDown(image)
    g_pyr.append(image)

  return g_pyr

def generate_laplacian_pyramid(image, levels):
  g_pyr = generate_gaussian_pyramid(image, levels)
  l_pyr = [g_pyr[-1]]
  for i in range(levels-1, 0 -1):
    g_expanded = cv2.pyrUp(g_pyr(i))
    g_expanded = spatial_lowpass(g_expanded)
    lap = cv2.subtract(g_pyr[i-1], g_expanded)
    l_pyr.append(lap)

  return l_pyr[-1]

def spatial_lowpass(image, kernel=np.ones((25,25), np.float32)/625):
  return cv2.filter2D(image, -1, kernel)


if __name__ == '__main__':
  stream = videoStream(0, 'CAFFE')
