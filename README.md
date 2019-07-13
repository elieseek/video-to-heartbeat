# video-to-heartbeat
An implementation of the paper: [Wu, Hao-Yu, et al. "Eulerian video magnification for revealing subtle changes in the world." (2012).](http://people.csail.mit.edu/mrub/papers/vidmag.pdf)

# Dependencies
Python packages:
* opencv-python
* numpy
* scipy
* matplotlib

Models for face detection need to be placed in src/pythonHeartbeatAnalyser folder. They are available from [here](https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models)
* deploy.prototext
* res10_300x300_ssd_iter_140000_fp16.caffemodel
* opencv_face_detector_uint8.pb
* opencv_face_detector.pbtext
# Current Functionality and limitations
* With a high quality webcam under perfect conditions (still subject) it can usually detect whether a heartbeat is present
* The heartbeat displayed is inaccurate

# Future work
- [ ] Experiment filtering using wavelets
- [ ] Implement in JavaScript with opencv.js
- [ ] Use multiple samples from faces to reduce noise
