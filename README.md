# Face_Recognition
* Used atleast 50 frames to train more accurately.
* Used basic cv2 methos i.e., casscade classifier and frame
* Used harrascade.frontal_face.xml file.

# Train_model_by_face_detection.py file

* Using harrascade_frontal_face.xml file
* x,y,w,h are coordinates of face
* Gray image is used because it act as a 2d image.
* Train about count number of frames.
* Saved file as .npy format.

# Face_Recognition.py File
* Using  KNeighborsClassifier as a model.
* Train data using Image.npy file.
* Flatten the data in the frames
* After that predict using model.predict

# Output

* Name of the person in front of camera whose face is trained.

# Can also visit for the same :  https://otaku-99.github.io/Face_Recognition/
