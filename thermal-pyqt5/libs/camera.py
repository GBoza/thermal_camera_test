# camera.py
# import the necessary packages
import cv2
import os
import time
import dlib
import numpy as np
from imutils import face_utils

face_landmark_path = './models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def get_pose_from_image(image):
    if image is None:
        return
    face_rects = detector(image, 0)
    if len(face_rects) > 0:
        for face_rect in face_rects:
            shape = predictor(image, face_rect)
            shape = face_utils.shape_to_np(shape)
            reprojectdst , euler_angle = get_head_pose(shape)
            pose_yaw = euler_angle[1,0]
            pose_pitch = euler_angle[0,0]
            pose_roll = euler_angle[2,0]
            return (pose_yaw, pose_pitch, pose_roll)
    return
        
def crop_face_from_image(image, face_rect):
    (x,y,x2,y2) = face_rect.left(),face_rect.top(),face_rect.right(),face_rect.bottom()
    face_img = image[y: y2, x: x2]
    return face_img

ds_factor = 0.5
# euler_angle = None


NUM_OF_PICTURES_RIGHT = 1
NUM_OF_PICTURES_LEFT = 1
NUM_OF_PICTURES_CENTER = 2 

PICTURES_DIR = './pictures'

class VideoCamera():
    def __init__(self):
        #capturing video
        # print("Constructor method of cam")
        time.sleep(1)
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.counter_frontal = {-2:0, -1:0 , 0:0, 1:0, 2:0 }
        self.counter_frontal_bool = {-2:False, -1:False , 0:False, 1:False, 2:False}
        #
        self.start_register_flag = False
        self.current_folder = ""

    def __del__(self):
        #releasing camera
        self.video.release()
    
    def get_frame(self):
        #extracting frames
        ret, fr_initial = self.video.read()
        fr_part_right = fr_initial[:, 640:]
        width = int(fr_part_right.shape[1] * ds_factor)                    
        height = int(fr_part_right.shape[0] * ds_factor)
        dim = (width, height)
        frame = cv2.resize(fr_part_right, dim)
        
        if ret:
            self.face_rects = detector(frame, 0)
            if len(self.face_rects) > 0:
                shape = predictor(frame, self.face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)
                # store frontal snaps

                if sum(list(self.counter_frontal.values()))== 0:
                    self.current_folder = self.create_folder_of_person()
                else:
                    self.current_folder = self.get_current_folder_of_person()

                self.frontal_face_snaps(euler_angle, frame, self.counter_frontal, self.counter_frontal_bool, self.current_folder)

                # Line and text draws, it is for test pourposes
                # for (x, y) in shape:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                # for start, end in line_pairs:
                #     cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                
                # cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)
                # cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)
                # cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def frontal_face_snaps(self, angles, frame, counter_frontal, counter_frontal_bool, file_path):
        """
        
        """
        if angles[1,0] > -25.0 and angles[1, 0] < -10.0:
            if counter_frontal[-2] < NUM_OF_PICTURES_RIGHT:
                cv2.imwrite(file_path + "pic_angle_y_{:.1f}.jpg".format(angles[1, 0]), frame)
                counter_frontal[-2] = counter_frontal[-2] + 1
            else:
                counter_frontal_bool[-2] = True
        
        elif angles[1,0] > -10.0 and angles[1, 0] < -5.0:
            if counter_frontal[-1] < NUM_OF_PICTURES_RIGHT:
                cv2.imwrite(file_path + "pic_angle_y_{:.1f}.jpg".format(angles[1, 0]), frame)
                counter_frontal[-1] = counter_frontal[-1] + 1
            else:
                counter_frontal_bool[-1] = True
        
        elif angles[1,0] > -5.0 and angles[1, 0] < 5.0:
            if counter_frontal[0] < NUM_OF_PICTURES_CENTER:
                cv2.imwrite(file_path + "pic_angle_y_{:.1f}.jpg".format(angles[1, 0]), frame)
                counter_frontal[0] = counter_frontal[0] + 1
            else:
                counter_frontal_bool[0] = True
        
        elif angles[1,0] > 5.0 and angles[1, 0] < 10.0:
            if counter_frontal[1] < NUM_OF_PICTURES_LEFT:
                cv2.imwrite(file_path + "pic_angle_y_{:.1f}.jpg".format(angles[1, 0]), frame)
                counter_frontal[1] = counter_frontal[1] + 1
            else:
                counter_frontal_bool[1] = True

        elif angles[1,0] > 10.0 and angles[1, 0] < 25.0:
            if counter_frontal[2] < NUM_OF_PICTURES_LEFT:    
                cv2.imwrite(file_path + "pic_angle_y_{:.1f}.jpg".format(angles[1, 0]), frame)
                counter_frontal[2] = counter_frontal[2] + 1
            else:
                counter_frontal_bool[2] = True
        else:
            pass
        print(counter_frontal_bool)
        print(counter_frontal)        


        


    def face_detected(self):
        if(len(self.face_rects) > 0):
            return True
        else:
            return False

    def get_status_of_pictures(self):
        v = self.counter_frontal.values()
        s = sum(v)
        return self.counter_frontal, s

    def get_flag_status_of_pictures(self):
        return (self.counter_frontal_bool[-2] and
        self.counter_frontal_bool[-1] and
        self.counter_frontal_bool[0] and
        self.counter_frontal_bool[1] and
        self.counter_frontal_bool[2])


