import cv2
import numpy as np
from threading import Thread

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

from objloader import *

from PIL import Image

# Modify these variables as needed
src = 0 # index for video capture function

# Window Settings
width, height = 1280, 720  # Window Size
x_coord, y_coord= 200, 200  # Window Position

#################### GLOBAL VARIABLES ####################
# Initialize camera capture and read first frame
cap = cv2.VideoCapture(src)
new_frame = cap.read()[1]

# Variables for threading
thread_quit = False

# GLUT Variables
texture_id = 0

# MUST RECALIBRATE FOR EACH CAMERA
# Camera matrix and distorition coefficients can be found via camera calibration (see camera_calib if you do not know these values)
mtx = np.array([  [955.,   0., 650.],
                  [  0., 955., 380.],
                  [  0.,   0.,   1.]])
dist = np.array([[0.14853128, -0.47652419, 0.00611533, -0.00105457, 0.63106111]])

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

##########################################################

def init_camera_capture():
    cap_thread = Thread(target=update, args=())
    cap_thread.start()

def update():
    global new_frame
    global thread_quit
    
    while(True):
        new_frame = cap.read()[1]

        if thread_quit:
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def draw():
    global new_frame
    global texture_id

    # Clear buffers to preset values
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glLoadIdentity()

    frame = new_frame # Update the frame

    glDisable(GL_DEPTH_TEST)

    # Convert frame to OpenGL texture format
    tx_img = cv2.flip(frame, 0)
    # tx_img = cv2.flip(tx_img, 1)
    tx_img = Image.fromarray(tx_img)
    ix, iy = tx_img.size
    tx_img = tx_img.tobytes('raw', 'BGRX', 0, -1)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -16.0)

    # Draw background
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0); glVertex3f(-8.0, -6.0, 0.0)
    glTexCoord2f(1.0, 1.0); glVertex3f( 8.0, -6.0, 0.0)
    glTexCoord2f(1.0, 0.0); glVertex3f( 8.0,  6.0, 0.0)
    glTexCoord2f(0.0, 0.0); glVertex3f(-8.0,  6.0, 0.0)
    glEnd()
    glPopMatrix()

    # Create texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_img)

    # Render Object
    glEnable(GL_DEPTH_TEST)
    track(frame)

    glutSwapBuffers()

def track(frame):
    global obj, obj2, obj3, obj4

    inverse = np.array([  [ 1.0, 1.0, 1.0, 1.0],
                          [-1.0,-1.0,-1.0,-1.0],
                          [-1.0,-1.0,-1.0,-1.0],
                          [ 1.0, 1.0, 1.0, 1.0]])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict,
                                              parameters=parameters,
                                              cameraMatrix=mtx,
                                              distCoeff=dist)
    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec
            if ids[i] == 1:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            elif ids[i] == 2:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            elif ids[i] == 3:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)
            else:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.5, mtx, dist)
            rmtx = cv2.Rodrigues(rvec)[0]

            view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvec[0,0,0]],
                                    [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvec[0,0,1]],
                                    [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvec[0,0,2]],
                                    [0.0       ,0.0       ,0.0       ,1.0    ]])
            view_matrix = view_matrix * inverse
            view_matrix = np.transpose(view_matrix)
            glPushMatrix()
            glLoadMatrixd(view_matrix)
            glRotate(90, 1, 0, 0)
            glRotate(90, 0, 1, 0)
            glTranslate(0.5, 1.3, 0.2)
            if ids[i] == 1:
                obj2.render()
            elif ids[i] == 2:
                obj3.render()
            elif ids[i] == 3:
                obj4.render()
            else:
                obj.render()
            glPopMatrix()
    
    cv2.imshow('Frame', frame)

def quit_key(key, x, y):
    # Convert bytes object to string 
    key = key.decode("utf-8")

    # Allows the window to be quitted out by pressing 'q'
    if key == "q":
        os._exit(1)

def init_gl():
    global texture_id
    global obj, obj2, obj3, obj4

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(33.7, 1.3, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
        
    # load object
    obj = OBJ('infiniti/infiniti.obj', swapyz=False)
    obj2 = OBJ('pickup/pickup.obj', swapyz=False)
    obj3 = OBJ("police/police.obj", swapyz=False)
    obj4 = OBJ("taxi/taxi.obj", swapyz=False)


    # assign texture
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)

if __name__ == "__main__":
    # Initialize camera capture thread
    init_camera_capture()

    # Initialize GLUT (Open Graphics Library Utility Toolkit)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(x_coord, y_coord)

    # Create a window
    window = glutCreateWindow('Render')

    # GLUT functions
    glutDisplayFunc(draw)
    glutIdleFunc(draw)
    glutKeyboardFunc(quit_key)  # Press "q" to quit out

    init_gl()

    glutMainLoop()