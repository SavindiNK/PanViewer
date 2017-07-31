import numpy as np
import cv2
import glob


class Calibrator:
    def __init__(self):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # image samples required for calibration
        self.image_path = glob.glob('C:\Users\Charith\Downloads\OpenCV 3.2\opencv\sources\samples\data\left??.jpg')
        # print self.image_path
        # calibrate matrix location
        self.calib_data_path = './calibration_data/calib.npz'

        # estimated calibrate parameters
        self.calibrated = False
        self.cam_matrix = None
        self.dis_coef = None
        self.rot_vector = None
        self.trans_vector = None

        # optimal camera matrix - depends on the frame size of the image
        self.optimal_cam_matrix = None
        self.roi = None

    def calibrate(self):
        # prepare object points
        obj_p = np.zeros((6 * 7, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # arrays to store object points
        obj_points = []
        img_points = []
        img_shape = None

        for image_file in self.image_path:
            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape
            # find chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # if found obj points, image points after refining them
            if ret == True:
                obj_points.append(obj_p)
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                img_points.append(corners)

        (self.calibrated, self.cam_matrix, self.dis_coef, self.rot_vector, self.trans_vector) = \
            cv2.calibrateCamera(obj_points, img_points, img_shape,None,None)

    def save_calibrate_data(self):
        if self.calibrated:
            # out_file = open(self.calib_data_path, 'w')
            np.savez(self.calib_data_path, cam_matrix=self.cam_matrix, dis_coef=self.dis_coef, rot_vector=self.rot_vector,
                     trans_vector=self.trans_vector)

    def load_from_memory(self):
        try:
            #in_file = open(self.calib_data_path, 'r')
            ar = np.load(self.calib_data_path)
            print ar.files
            # self.trans_vector = ar['trans_vector']
            # self.cam_matrix = ar['cam_matrix']
            # self.dis_coef = ar['dis_coef']
            # self.rot_vector = ar['rot_vector']

            self.calibrated = True
            return True
        except IOError:
            print 'Error: Load from memory failed!'
            return False

    # image width and height should be passed in order to calculate this
    def calculate_optimal_camera_matrix(self, w, h):
        self.optimal_cam_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.dis_coef, (w,h), 1, (w,h))


class Distort_Remover:
    def __init__(self, calibrator):
        self.calibrator = calibrator

    def undistort(self, image):
        if self.calibrator.optimal_cam_matrix is not None:
            dst = cv2.undistort(image, self.calibrator.cam_matrix, self.calibrator.dis_coef, None, self.calibrator.optimal_cam_matrix)
            x, y, w, h = self.calibrator.roi
            return dst[y:y+h, x:x+w]
        else:
            return None