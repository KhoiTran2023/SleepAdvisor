import time

import cv2
import dlib
import numpy as np

import flet as f

from Utils import get_face_area, lip_distance, add_alert
from Eye_Dector_Module import EyeDetector as EyeDets
from Attention_Scorer_Module import AttentionScorer as AttScorer

from imutils import face_utils

# capture source number select the webcam to use (default is zero -> built in camera)
CAPTURE_SOURCE = 0
# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array([[1806.85434, 0.00000000, 1017.00291],[0.00000000, 1791.41030, 837.096481],[0.00000000, 0.00000000, 1.00000000]], dtype = "double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.29718555, 0.16997364,-0.0041329,-0.00174958,0.04838176]], dtype="double")


def main():

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit
    fps_lim = 7  # FPS upper limit value, needed for estimating the time for each frame and increasing performances
    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

    cv2.setUseOptimized(True)  # set OpenCV optimization to True
    # instantiation of the dlib face detector object
    detector = cv2.CascadeClassifier("predictor/haarcascade_frontalface_default.xml")    #Faster but less accurate
    predictor = dlib.shape_predictor(
    "predictor/shape_predictor_68_face_landmarks.dat")

    Detector = dlib.get_frontal_face_detector()
    '''
    the keypoint predictor is compiled in C++ and saved as a .dat inside the "predictor" folder in the project
    inside the folder there is also a useful face keypoint image map to understand the position and numnber of the
    various predicted face keypoints
    '''
    # instantiation of the eye detector and pose estimator objects
    Eye_det = EyeDets(show_processing=False)


    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    Scorer = AttScorer(fps_lim, ear_tresh=0.15, ear_time_tresh=2, gaze_tresh=0.2,
                       gaze_time_tresh=2, verbose=False)
    
    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(CAPTURE_SOURCE)

    gazeCounter = 0
    tiredCounter = 0
    yawnCounter = 0
    yawnTresh = 4
    perclos_tresh = 0.15
    level_three_warning = 0
    level_two_warning = int(fps_lim*120*perclos_tresh)

    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()
    
    while True:  # infinite loop for webcam video capture
        delta_time = time.perf_counter() - prev_time  # delta time for FPS capping
        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

         # if the frame comes from webcam, flip it so it looks like a mirror.
        if CAPTURE_SOURCE == 0:
            frame = cv2.flip(frame, 2)

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.perf_counter()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime
            cv2.putText(frame, "FPS:" + str(round(fps,0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 255), 1)

            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply a bilateral filter to lower noise but keep frame details
            gray = cv2.bilateralFilter(gray, 5, 10, 10)

            # find the faces using the dlib face detector
            faces = Detector(gray)

            #yawn
            rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		    minNeighbors=5, minSize=(30, 30),
		    flags=cv2.CASCADE_SCALE_IMAGE)

            #for rect in rects:
            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
                
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                distance = lip_distance(shape)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 0, 255), 1)


                #yawn evaluation
                if (distance > 27):
                    yawnCounter += 1
                else:
                    if(yawnCounter >= fps_lim*yawnTresh):
                        print("you yawned!")
                        level_three_warning +=1
                        yawnCounter = 0

            if len(faces) > 0:  # process the frame only if at least a face is found

                # take only the bounding box of the biggest face
                faces = sorted(faces, key=get_face_area, reverse=True)
                driver_face = faces[0]

                # predict the 68 facial keypoints position
                landmarks = predictor(gray, driver_face)

                # shows the eye keypoints (can be commented)
                Eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks)

                # compute the EAR score of the eyes
                ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

                # compute the PERCLOS score and state of tiredness
                tired, perclos_score = Scorer.get_PERCLOS(ear)

                # compute the Gaze Score
                gaze = Eye_det.get_Gaze_Score(frame=gray, landmarks=landmarks)
                
                # show the real-time EAR score
                if ear is not None:
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time Gaze Score
                if gaze is not None:
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time PERCLOS score
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                
                # if the driver is tired, show and alert on screen
                if tired:  
                    cv2.putText(frame, "TIRED!", (10, 280),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # evaluate the scores for EAR, GAZE
                asleep = Scorer.eval_scores(
                    ear, gaze)  

                # if the state of attention of the driver is not normal, show an alert on screen
                if asleep:
                    cv2.putText(frame, "ASLEEP!", (10, 300),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    print("CRITICAL WARNING: YOU ARE ASLEEP")
                
                
                if tired or perclos_score>= perclos_tresh:
                    tiredCounter+=1
                    if tiredCounter>level_two_warning:
                        level_three_warning += 1

                if gaze is None:
                    if gazeCounter>=fps_lim:
                        gazeCounter=0
                        print("Danger, eyes closed!")
                    gazeCounter+=1

                if level_three_warning >= 3 and level_three_warning < 4:
                    print("please stop driving immediately! you are tired.")
                    level_three_warning += 1
                elif level_three_warning > 0 and level_three_warning < 2:
                    print("careful! you may be tired")
                    level_three_warning += 1
            cv2.imshow("Frame", frame)  # show the frame on screen

            # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()