import time
import openai
import cv2
import dlib
import numpy as np
import psycopg2
import os
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

openai.api_key = os.getenv("OPEN_API_KEY")

def main(page: f.Page):
    def submit_func(e):
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a health analyst."},
            {"role": "user", "content": f"I work as a construction worker. In the past seven days, my sleep hours, work hours, and tiredness (Y/N) data follows: day 1 - {sleep1.value} hours, {work1.value} hours, {tired1.value}; day 2 - {sleep2.value} hours, {work2.value} hours, {tired2.value};day 3 - {sleep3.value} hours, {work3.value} hours, {tired3.value};day 4 - {sleep4.value} hours, {work4.value} hours, {tired4.value};day 5 - {sleep5.value} hours, {work5.value} hours, {tired5.value};day 6 - {sleep6.value} hours, {work6.value} hours, {tired6.value};day 7 - {sleep7.value} hours, {work7.value} hours, {tired7.value};Based on this data, is it a high probability being stressed because of my sleep or my work hours? If it is more sleep-weighted, give me advice on what to change within the next seven days and add at the end 'implement these for seven days and see how your results differ'. If it is more work-weighted, then tell me to speak with my employer, give me some advice, and at the end add 'your employer has been notified of this data'."},        ])
        a = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a health analyst."},
            {"role": "user", "content": f"I work as a construction worker. In the past seven days, my sleep hours, work hours, and tiredness (Y/N) data follows: day 1 - {sleep1.value} hours, {work1.value} hours, {tired1.value}; day 2 - {sleep2.value} hours, {work2.value} hours, {tired2.value};day 3 - {sleep3.value} hours, {work3.value} hours, {tired3.value};day 4 - {sleep4.value} hours, {work4.value} hours, {tired4.value};day 5 - {sleep5.value} hours, {work5.value} hours, {tired5.value};day 6 - {sleep6.value} hours, {work6.value} hours, {tired6.value};day 7 - {sleep7.value} hours, {work7.value} hours, {tired7.value};Based on this data, is it a high probability being stressed because of my sleep or my work hours? based on this data, is it a high probability being stressed because of my sleep or my work hours? Answer in one-word: 'sleep' or 'work', I don't want anything else other than one word, no complete sentences just one word no quotation marks nothing just the word"}
        ])
        hours_slept = int(sleep1.value)+int(sleep2.value)+int(sleep1.value)+int(sleep1.value)+int(sleep1.value)+int(sleep1.value)+int(sleep1.value)
        hours_worked = int(work1.value)+int(work2.value)+int(work3.value)+int(work4.value)+int(work5.value)+int(work6.value)+int(work7.value)
        conn = psycopg2.connect(database = "sleepadvisor", 
                        user = "sleepadvisor_user", 
                        host= 'oregon-postgres.render.com',
                        password = "Sp96b62g9malQTDu0DFLoOwWOMhB9W0B",
                        port = 5432)
        cur = conn.cursor()
        cur.execute(f"INSERT INTO employee_data(hours_slept, hours_worked, feedback) VALUES({hours_slept},{hours_worked},'{a.choices[0].message['content']}')");
        conn.commit()
        cur.execute('SELECT * FROM employee_data;')
        rows = cur.fetchall()
        conn.commit()
        cur.close()
        conn.close()
        for row in rows:
            print(row)
        add_alert(page, l, f"Based on your results in the past seven days our analytics show: {completion.choices[0].message['content']}.")
    page.title = "SleepAdvisor - Corporate Corporate Well-Being"
    tired1 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )
    tired2 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )
    tired3 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )
    tired4 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )
    tired5 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )
    tired6 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )
    tired7 = f.Dropdown(
        width=100,
        options=[
            f.dropdown.Option("Y"),
            f.dropdown.Option("N"),
        ],
    )

    sleep1 = f.TextField(label = "Hours slept")
    sleep2 = f.TextField(label = "Hours slept")
    sleep3 = f.TextField(label = "Hours slept")
    sleep4 = f.TextField(label = "Hours slept")
    sleep5 = f.TextField(label = "Hours slept")
    sleep6 = f.TextField(label = "Hours slept")
    sleep7 = f.TextField(label = "Hours slept")

    work1 = f.TextField(label = "Hours worked")
    work2 = f.TextField(label = "Hours worked")
    work3 = f.TextField(label = "Hours worked")
    work4 = f.TextField(label = "Hours worked")
    work5 = f.TextField(label = "Hours worked")
    work6 = f.TextField(label = "Hours worked")
    work7 = f.TextField(label = "Hours worked")
    page.add(
        f.DataTable(
            columns = [
                f.DataColumn(f.Text("Day #")),
                f.DataColumn(f.Text("Tired? (Y/N)")),
                f.DataColumn(f.Text("Hours Slept?")),
                f.DataColumn(f.Text("Hours Worked?"))
            ],
            rows = [
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("1")),
                        f.DataCell(tired1),
                        f.DataCell(sleep1),
                        f.DataCell(work1),
                    ],
                ),
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("2")),
                        f.DataCell(tired2),
                        f.DataCell(sleep2),
                        f.DataCell(work2),
                    ],
                ),
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("3")),
                        f.DataCell(tired3),
                        f.DataCell(sleep3),
                        f.DataCell(work3),
                    ],
                ),
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("4")),
                        f.DataCell(tired4),
                        f.DataCell(sleep4),
                        f.DataCell(work4),
                    ],
                ),
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("5")),
                        f.DataCell(tired5),
                        f.DataCell(sleep5),
                        f.DataCell(work5),
                    ],
                ),
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("6")),
                        f.DataCell(tired6),
                        f.DataCell(sleep6),
                        f.DataCell(work6),
                    ],
                ),
                f.DataRow(
                    cells=[
                        f.DataCell(f.Text("7")),
                        f.DataCell(tired7),
                        f.DataCell(sleep7),
                        f.DataCell(work7),
                    ],
                ),
            ]
        )
    )
    submit_btn = f.ElevatedButton(text="Submit", on_click = submit_func)
    page.add(submit_btn, f.Text("Analytics Feedback:", size=20))
    l = f.ListView(expand=1, spacing=0, padding=10, auto_scroll=True)
    page.add(l)
    page.add(f.Text("Fatigue Detector Indicator Below, show your face:", size = 20))
    lv = f.ListView(expand=1, spacing=0, padding=10, auto_scroll=True)

    page.add(lv)
    add_alert(page, lv, "Nothing so far. You're nice and cheery!")

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
                        add_alert(page, lv, "You yawned!")
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

                if tired or perclos_score>= perclos_tresh:
                    tiredCounter+=1
                    if tiredCounter>level_two_warning:
                        level_three_warning += 1

                if gaze is None:
                    if gazeCounter>=fps_lim:
                        gazeCounter=0
                    gazeCounter+=1

                if level_three_warning >= 3 and level_three_warning < 4:
                    print("you are tired.")
                    add_alert(page, lv, "you are tired.")
                    level_three_warning += 1
                elif level_three_warning > 0 and level_three_warning < 2:
                    print("careful! you may be tired")
                    add_alert(page, lv, "Careful! You may be tired.")
                    level_three_warning += 1
    cap.release()
    cv2.destroyAllWindows()

    return

f.app(target=main)