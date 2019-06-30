"use strict";

const FPS = 30

var src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
var dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
var gray = new cv.Mat();
var cap = new cv.VideoCapture(video);
var faces = new cv.RectVector();

var streaming = false;
var inputVideo = document.getElementById('inputVideo')
var videoToggle = document.getElementById('videoToggle')
var outputCanvas = document.getElementById('outputCanvas')
var canvasContext = outputCanvas.getContext('2d')

var userData = {
  'frame': null,
  'b': null,
  'g': null,
  'r': null,
  'framesSinceUpdate': 0,
  'rawSignal': null,
  'concattedSignal': null,
  'bpm': 'calculating bpm',
  'hsvImg': null
};

function analyseVideo() {
  try {
    if (!streaming) {
      src.delete();
      dst.delete();
      gray.delete();
      faces.delete();
      return;
    }
  } catch (err) {
    console.log(err)
  }
}

videoToggle.addEventListener('click', () => {
  if (!streaming) {
    
  }
});

