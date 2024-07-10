from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import datetime
import time
from numba import jit  # , float32, int32
from statistics import mode
import cv2


# TODO: replace numpy operation like mean, std, median, diff etc. with numba direct calculations

@jit(nopython=True)
def calculate_mean_probability(labels, probabilities, majority_label):
    if majority_label == 'Unknown':
        # If majority is unknown then we return probability np.nan
        return np.nan

    sum_probabilities_majority_label = 0
    count_probabilities_majority_label = 0

    for i in range(len(labels)):
        label = labels[i]
        probability = probabilities[i]

        if label == majority_label:
            sum_probabilities_majority_label += probability
            count_probabilities_majority_label += 1

    return np.round((sum_probabilities_majority_label / count_probabilities_majority_label),
                    2) if count_probabilities_majority_label > 0 else np.nan


@jit(nopython=True)
def mean_without_zeros(data):
    total = 0
    count = 0
    for row in data:
        for value in row:
            if value != 0:
                total += value
                count += 1
    return int(total / count) if count > 0 else np.nan


@jit(nopython=True)
def calcstd(x, y):
    stdX = int(np.std(x))
    stdY = int(np.std(y))
    absStd = int(np.sqrt(np.square(stdX) + np.square(stdY)))

    return stdX, stdY, absStd


@jit(nopython=True)
def calcpathlength(x, y):
    xDiff = np.diff(x)
    yDiff = np.diff(y)
    pathLength = int(np.sum(np.sqrt(np.square(xDiff) + np.square(yDiff))))

    return pathLength


@jit(nopython=True)
def calcLocalVelocitySTD(Vx, Vy):
    n = len(Vx)
    if n > 1:
        mean_vx = 0.0
        mean_vy = 0.0

        # Calculate the mean velocities
        for vx, vy in zip(Vx, Vy):
            mean_vx += vx
            mean_vy += vy

        mean_vx /= n
        mean_vy /= n

        # Calculate the squared deviations from the mean
        squared_diffs = 0.0
        for vx, vy in zip(Vx, Vy):
            squared_diffs += (vx - mean_vx) ** 2 + (vy - mean_vy) ** 2

        # Calculate the standard deviation
        v_std = np.sqrt(squared_diffs / n)

        return v_std

    else:

        return 0


def is_any_value_in_list(values_to_check, input_list):
    return bool(any(value in input_list for value in values_to_check))


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Function to compute pairwise IoU
def pairwise_iou(boxesA, boxesB):
    iou_matrix = np.zeros((len(boxesA), len(boxesB)))
    for i, boxA in enumerate(boxesA):
        for j, boxB in enumerate(boxesB):
            iou_matrix[i, j] = compute_iou(boxA, boxB)
    return iou_matrix


# Function to compute combined metric
def combined_metric(centroidsA, centroidsB, boxesA, boxesB, alpha):
    euclidean_distances = dist.cdist(centroidsA, centroidsB, metric='euclidean')
    max_dist = np.max(euclidean_distances)
    if max_dist == 0:
        normalized_distances = euclidean_distances
    else:
        normalized_distances = euclidean_distances / max_dist
    iou_matrix = pairwise_iou(boxesA, boxesB)
    combined_scores = alpha * normalized_distances + (1 - alpha) * (1 - iou_matrix)
    return euclidean_distances, combined_scores


class CentroidTracker:
    def __init__(self, nextObjectID, maxDisappearedTime, maxDistance, minLifeTime, maxNewDetections,
                 tensorrt_model, classConfidence, minTimeToStartInference,
                 inferenceFrequency, maxClipLengthMin, majorityVoting,
                 use_unscaled_frame_for_inference, width, height, use_xgboost_classification_modality,
                 xgboost, IoU_dist_alpha_blend):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = nextObjectID

        self.sensorID = OrderedDict()
        self.centroids = OrderedDict()
        self.centroidsMedian = OrderedDict()
        self.boundingBoxs = OrderedDict()
        self.area = OrderedDict()
        self.velocity = OrderedDict()
        self.meanAbsVelocity = OrderedDict()
        self.velocity_m1 = OrderedDict()
        self.velocitySTD = OrderedDict()
        self.frameTime = OrderedDict()
        self.aspectRatio = OrderedDict()
        self.deletedEventsIDs = []
        self.events = []
        self.newDetections = 0
        self.std = OrderedDict()
        self.absStd = OrderedDict()
        self.pathLength = OrderedDict()
        self.xAngle = OrderedDict()
        self.yAngle = OrderedDict()
        self.pathAngle = OrderedDict()

        self.R = OrderedDict()
        self.G = OrderedDict()
        self.B = OrderedDict()

        self.startFrame = OrderedDict()
        self.endFrame = OrderedDict()
        self.startDisappearedTime = OrderedDict()
        self.disappearedTime = OrderedDict()
        self.lifeTime = OrderedDict()
        self.startTime = OrderedDict()
        self.endTime = OrderedDict()

        self.meanAbsVelocity_path_history = OrderedDict()
        self.vX_path_history = OrderedDict()
        self.vY_path_history = OrderedDict()
        self.velocitySTD_path_history = OrderedDict()
        self.aspectRatio_path_history = OrderedDict()
        self.stdX_path_history = OrderedDict()
        self.stdY_path_history = OrderedDict()
        self.absStd_path_history = OrderedDict()
        self.lifeTime_path_history = OrderedDict()
        self.sqrt_area_path_history = OrderedDict()
        self.pathLength_path_history = OrderedDict()

        self.objectID_path_history_final = []
        self.bboxPath = {}
        self.meanAbsVelocity_path_history_final = {}
        self.vX_path_history_final = {}
        self.vY_path_history_final = {}
        self.velocitySTD_path_history_final = {}
        self.aspectRatio_path_history_final = {}
        self.stdX_path_history_final = {}
        self.stdY_path_history_final = {}
        self.absStd_path_history_final = {}
        self.lifeTime_path_history_final = {}
        self.sqrt_area_path_history_final = {}
        self.pathLength_path_history_final = {}

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappearedTime = maxDisappearedTime
        self.maxDistance = maxDistance
        self.minVisibleTime = minLifeTime
        self.maxNewDetections = maxNewDetections

        self.maxClipLengthSec = maxClipLengthMin * 60 - 2  # Reduce 2 seconds due to clip margins

        self.label = OrderedDict()
        self.labelProbability = OrderedDict()
        self.classesProbabilities = OrderedDict()
        self.classProbMeanCounter = OrderedDict()
        self.object_features = OrderedDict()

        self.cropped_images = OrderedDict()

        self.classConfidence = classConfidence
        self.minTimeToStartInference = minTimeToStartInference
        self.inferenceFrequency = inferenceFrequency

        self.updateVelocityFlag = True
        self.updateAreaAspectRatioFlag = True
        self.updateRGBFlag = True
        self.updatePathAngleFlag = True
        self.updatexyAngleFlag = True
        self.updateSTDflag = True
        self.updatePathLengthFlag = True
        self.imageClassificationFlag = False

        if self.imageClassificationFlag:
            self.trt = tensorrt_model
            self.numOfClasses = len(self.trt.labels)
        else:
            self.numOfClasses = 0

        self.majorityVoting = majorityVoting

        self.use_unscaled_frame_for_inference = use_unscaled_frame_for_inference
        self.width = width
        self.height = height
        self.full_frame = None

        # with open('imageResizeValue.txt', 'r') as file:
        #     dim = file.readline().strip()  # Read the first line and remove any whitespace characters
        #     min_dim = file.readline().strip()  # Read the second line
        #     max_dim = file.readline().strip()  # Read the third line

        self.image_resize = (128, 128)  # (int(dim), int(dim))
        self.min_dim = 15  # int(min_dim)
        self.max_dim = 400  # int(max_dim)

        self.use_xgboost_classification_modality = use_xgboost_classification_modality
        self.xgboost = xgboost

        self.IoU_dist_alpha_blend = IoU_dist_alpha_blend

    def register(self, centroid, bbox, frameID, sensorID, area):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.sensorID[self.nextObjectID] = sensorID[0]
        self.centroids[self.nextObjectID] = centroid
        self.boundingBoxs[self.nextObjectID] = [np.append(bbox, frameID).tolist()]
        self.centroidsMedian[self.nextObjectID] = [centroid[0], centroid[1], [centroid[0]],
                                                   [centroid[1]]]  # [cX, cY, cXData, cYData]

        self.startTime[self.nextObjectID] = time.time()
        self.endTime[self.nextObjectID] = time.time()
        self.startDisappearedTime[self.nextObjectID] = time.time()
        self.lifeTime[self.nextObjectID] = 0
        self.disappearedTime[self.nextObjectID] = 0
        self.startFrame[self.nextObjectID] = frameID
        self.endFrame[self.nextObjectID] = frameID

        bbWidth = bbox[2] - bbox[0]
        bbHeight = bbox[3] - bbox[1]
        # area = bbWidth * bbHeight
        # area = area[0]
        self.area[self.nextObjectID] = [[], np.nan]

        aspectRatio = bbHeight / bbWidth
        self.aspectRatio[self.nextObjectID] = [[], np.nan]

        self.frameTime[self.nextObjectID] = time.time()
        self.velocity[self.nextObjectID] = [np.nan, np.nan, [], []]  # [pixels/sec] [vX, Vy, VxData, VyData]
        self.meanAbsVelocity[self.nextObjectID] = np.nan  # [pixels/sec]
        self.velocity_m1[self.nextObjectID] = np.nan
        self.velocitySTD[self.nextObjectID] = np.nan

        self.meanAbsVelocity_path_history[self.nextObjectID] = []
        self.vX_path_history[self.nextObjectID] = []
        self.vY_path_history[self.nextObjectID] = []
        self.velocitySTD_path_history[self.nextObjectID] = []
        self.aspectRatio_path_history[self.nextObjectID] = []
        self.stdX_path_history[self.nextObjectID] = []
        self.stdY_path_history[self.nextObjectID] = []
        self.absStd_path_history[self.nextObjectID] = []
        self.lifeTime_path_history[self.nextObjectID] = []
        self.sqrt_area_path_history[self.nextObjectID] = []
        self.pathLength_path_history[self.nextObjectID] = []

        self.R[self.nextObjectID] = []
        self.G[self.nextObjectID] = []
        self.B[self.nextObjectID] = []

        self.xAngle[self.nextObjectID] = [np.nan, []]  # [cos direction] [current direction, accumulated data]
        self.yAngle[self.nextObjectID] = [np.nan, []]  # [sin direction] [current direction, accumulated data]
        self.pathAngle[self.nextObjectID] = [np.nan, []]  # path angle

        self.std[self.nextObjectID] = [np.nan, np.nan]  # [stdX, stdY]
        self.absStd[self.nextObjectID] = np.nan
        self.pathLength[self.nextObjectID] = np.nan

        self.label[self.nextObjectID] = ['', []]  # [majority voting label, [accumulated labels]]
        self.labelProbability[self.nextObjectID] = [np.nan,
                                                    []]  # [mean majority voting probability, [accumulated probabilities]]

        self.object_features[self.nextObjectID] = [np.nan, []]  # [features_array, [accumulated features array]]

        self.classesProbabilities[self.nextObjectID] = [np.nan] * self.numOfClasses

        self.classProbMeanCounter[self.nextObjectID] = 0

        self.cropped_images[self.nextObjectID] = []

        self.nextObjectID += 1

        self.newDetections += 1

    def buildEventRow(self, objectID):

        now = datetime.datetime.now()

        minutes_since_midnight = now.hour * 60 + now.minute

        # TODO for start/end position calc the mean or median of about 0.5/15 frames from the start or from the end of event (to avoid jumps in poisition due to outlayers)
        startLocationX = self.centroidsMedian[objectID][2][0]
        endLocationX = self.centroidsMedian[objectID][2][-1]

        startLocationY = self.centroidsMedian[objectID][3][0]
        endLocationY = self.centroidsMedian[objectID][3][-1]

        if self.velocity[objectID][2] and self.velocity[objectID][3]:
            maxVelocity = int(np.max([np.abs(self.velocity[objectID][2]) + np.abs(self.velocity[objectID][3])]))
        else:
            maxVelocity = np.nan

        falseAlarmProbability = 0
        isAnnomaly = 0

        eventRow = [objectID,  # ID
                    self.sensorID[objectID],  # sensor ID
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    np.nan,  # Retro score - computed later
                    np.nan,  # Score - computed later
                    isAnnomaly,  # Anomaly Detection
                    '',  # User Feedback
                    self.label[objectID][0],  # majority voting Label
                    self.labelProbability[objectID][0],  # majority voting mean probability
                    '',  # User Class
                    0,  # Model Ignore (0 - not ignore in model, 1 - ignore in model)
                    np.round(self.lifeTime[objectID], 1),  # life Time [sec]
                    self.startFrame[objectID],  # start frame
                    self.endFrame[objectID],  # end frame
                    self.centroidsMedian[objectID][0],  # median cX
                    self.centroidsMedian[objectID][1],  # median cY
                    self.velocity[objectID][0],  # vX [pixels/sec]
                    self.velocity[objectID][1],  # vY [pixels/sec]
                    self.meanAbsVelocity[objectID],  # mean absolute velocity [pixels/sec]
                    self.velocity_m1[objectID],  # 1 / meanAbsVelocity
                    self.velocitySTD[objectID],  # standard deviation of local velocity
                    self.pathAngle[objectID][0],  # path angle
                    self.xAngle[objectID][0],  # x direction angle
                    self.yAngle[objectID][0],  # y direction angle
                    np.nan if not self.R[objectID] else int(np.mean(self.R[objectID])),
                    np.nan if not self.G[objectID] else int(np.mean(self.G[objectID])),
                    np.nan if not self.B[objectID] else int(np.mean(self.B[objectID])),
                    np.round(self.area[objectID][1], 1),  # median area
                    self.aspectRatio[objectID][1],  # median aspect ratio
                    startLocationX,
                    endLocationX,
                    startLocationY,
                    endLocationY,
                    self.pathLength[objectID],
                    self.std[objectID][0],  # stdX
                    self.std[objectID][1],  # stdY
                    self.absStd[objectID],
                    maxVelocity,
                    minutes_since_midnight,
                    now.hour,
                    now.weekday(),
                    ]

        if self.numOfClasses > 1 and not self.majorityVoting:
            eventRow.extend(self.classesProbabilities[objectID])

        return eventRow

    def deregister(self, objectID, frameID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries

        self.endFrame[objectID] = frameID

        self.objectID_path_history_final.append(objectID)
        self.bboxPath[objectID] = self.boundingBoxs[objectID]
        self.meanAbsVelocity_path_history_final[objectID] = self.meanAbsVelocity_path_history[objectID]
        self.vX_path_history_final[objectID] = self.vX_path_history[objectID]
        self.vY_path_history_final[objectID] = self.vY_path_history[objectID]
        self.velocitySTD_path_history_final[objectID] = self.velocitySTD_path_history[objectID]
        self.aspectRatio_path_history_final[objectID] = self.aspectRatio_path_history[objectID]
        self.stdX_path_history_final[objectID] = self.stdX_path_history[objectID]
        self.stdY_path_history_final[objectID] = self.stdY_path_history[objectID]
        self.absStd_path_history_final[objectID] = self.absStd_path_history[objectID]
        self.lifeTime_path_history_final[objectID] = self.lifeTime_path_history[objectID]
        self.sqrt_area_path_history_final[objectID] = self.sqrt_area_path_history[objectID]
        self.pathLength_path_history_final[objectID] = self.pathLength_path_history[objectID]

        temp_event_row = self.buildEventRow(objectID)
        temp_event_row.append(self.object_features[objectID][0])
        temp_event_row.append(self.cropped_images[objectID])

        self.events.append(temp_event_row)

        del self.centroids[objectID]
        del self.boundingBoxs[objectID]
        del self.centroidsMedian[objectID]
        del self.disappearedTime[objectID]
        del self.lifeTime[objectID]
        del self.area[objectID]
        del self.velocity[objectID]
        del self.pathAngle[objectID]
        del self.xAngle[objectID]
        del self.yAngle[objectID]
        del self.R[objectID]
        del self.G[objectID]
        del self.B[objectID]
        del self.aspectRatio[objectID]
        del self.startFrame[objectID]
        del self.endFrame[objectID]
        del self.sensorID[objectID]
        del self.std[objectID]
        del self.absStd[objectID]
        del self.pathLength[objectID]
        del self.startTime[objectID]
        del self.endTime[objectID]
        del self.startDisappearedTime[objectID]
        del self.meanAbsVelocity[objectID]
        del self.frameTime[objectID]
        del self.velocity_m1[objectID]
        del self.label[objectID]
        del self.object_features[objectID]
        del self.labelProbability[objectID]
        del self.velocitySTD[objectID]
        del self.classesProbabilities[objectID]
        del self.classProbMeanCounter[objectID]
        del self.cropped_images[objectID]

        del self.meanAbsVelocity_path_history[objectID]
        del self.vX_path_history[objectID]
        del self.vY_path_history[objectID]
        del self.velocitySTD_path_history[objectID]
        del self.aspectRatio_path_history[objectID]
        del self.stdX_path_history[objectID]
        del self.stdY_path_history[objectID]
        del self.absStd_path_history[objectID]
        del self.lifeTime_path_history[objectID]
        del self.sqrt_area_path_history[objectID]
        del self.pathLength_path_history[objectID]

        self.deletedEventsIDs.append(objectID)

    def updateRGB(self, objectID, cropped_object):

        if self.lifeTime[objectID] > self.minVisibleTime:
            # Cropping the image more to get only it's center part
            # m = 10  # [pixels] margin removal size
            # center_region = cropped_object[m:-m, m:-m, :]

            height, width, _ = cropped_object.shape
            center_y, center_x = height // 2, width // 2
            half_size = 5  # Half of the size of the center region (10x10 center region has half size 5)

            # Calculate the boundaries for the center region
            top = max(center_y - half_size, 0)
            bottom = min(center_y + half_size + 1, height)
            left = max(center_x - half_size, 0)
            right = min(center_x + half_size + 1, width)

            # Extract the center region or the largest available region
            center_region = cropped_object[top:bottom, left:right, :]

            mean_roi_B = mean_without_zeros(center_region[:, :, 0])
            mean_roi_G = mean_without_zeros(center_region[:, :, 1])
            mean_roi_R = mean_without_zeros(center_region[:, :, 2])

            if not np.isnan(mean_roi_B):
                self.B[objectID].append(mean_roi_B)

            if not np.isnan(mean_roi_G):
                self.G[objectID].append(mean_roi_G)

            if not np.isnan(mean_roi_R):
                self.R[objectID].append(mean_roi_R)

            # mean_values = (self.B[objectID][-1], self.G[objectID][-1], self.R[objectID][-1])
            # print('object ID', objectID, "Median values (B, G, R):", mean_values)

    def updateAreaAspectRatioMedian(self, objectID, currentArea):
        bbWidth = self.boundingBoxs[objectID][-1][2] - self.boundingBoxs[objectID][-1][0]
        bbHeight = self.boundingBoxs[objectID][-1][3] - self.boundingBoxs[objectID][-1][1]
        # currentArea = bbWidth * bbHeight
        currentAspectRatio = bbHeight / bbWidth

        self.area[objectID][0].append(currentArea)
        self.area[objectID][1] = np.median(self.area[objectID][0])

        self.aspectRatio[objectID][0].append(currentAspectRatio)
        self.aspectRatio[objectID][1] = np.round(np.median(self.aspectRatio[objectID][0]), 1)

    def updateCentroidsMedian(self, objectID):
        self.centroidsMedian[objectID][2].append(self.centroids[objectID][0])
        self.centroidsMedian[objectID][3].append(self.centroids[objectID][1])
        self.centroidsMedian[objectID][0] = np.median(self.centroidsMedian[objectID][2])
        self.centroidsMedian[objectID][1] = np.median(self.centroidsMedian[objectID][3])

        # stdX = np.round(np.std(self.centroidsMedian[objectID][2]))
        # stdY = np.round(np.std(self.centroidsMedian[objectID][3]))
        # absStd = np.round(np.sqrt((stdX ** 2 + stdY ** 2)))

        if self.updateSTDflag:
            stdX, stdY, absStd = calcstd(np.array(self.centroidsMedian[objectID][2]),
                                         np.array(self.centroidsMedian[objectID][3]))

            self.std[objectID] = [stdX, stdY]
            self.absStd[objectID] = absStd

        if self.updatePathLengthFlag:
            if len(self.centroidsMedian[objectID][2]) > 1:  # calc path length only after movement
                # xDiff = np.diff(self.centroidsMedian[objectID][2])
                # yDiff = np.diff(self.centroidsMedian[objectID][3])
                # self.pathLength[objectID] = np.round(np.sum(np.sqrt(xDiff ** 2 + yDiff ** 2)))

                self.pathLength[objectID] = calcpathlength(np.array(self.centroidsMedian[objectID][2]),
                                                           np.array(self.centroidsMedian[objectID][3]))

    def updateVelocity(self, objectID, previousCentroids):

        dT = time.time() - self.frameTime[objectID]
        self.frameTime[objectID] = time.time()

        vx = (self.centroids[objectID][0] - previousCentroids[0]) / dT
        vy = (self.centroids[objectID][1] - previousCentroids[1]) / dT

        self.velocity[objectID][2].append(vx)
        self.velocity[objectID][3].append(vy)

        self.velocity[objectID][0] = np.round(
            (self.centroidsMedian[objectID][2][-1] - self.centroidsMedian[objectID][2][0]) / self.lifeTime[objectID], 1)
        self.velocity[objectID][1] = np.round(
            (self.centroidsMedian[objectID][3][-1] - self.centroidsMedian[objectID][3][0]) / self.lifeTime[objectID], 1)

        absVelocity = np.sqrt(np.square(vx) + np.square(vy))
        vecLength = len(self.velocity[objectID][2])

        if np.isnan(self.meanAbsVelocity[objectID]):
            self.meanAbsVelocity[objectID] = round(absVelocity)
        else:
            self.meanAbsVelocity[objectID] = round(
                (self.meanAbsVelocity[objectID] * (vecLength - 1) + absVelocity) / vecLength)

        self.velocitySTD[objectID] = int(
            calcLocalVelocitySTD(np.array(self.velocity[objectID][2]), np.array(self.velocity[objectID][3])))

        # Utilizing int(1/(velocity+10) * 10000) serves the purpose of preventing issues when velocity is zero,
        # and the additional scaling by 10000 is implemented to facilitate integer operations while preserving a
        # satisfactory level of resolution.
        self.velocity_m1[objectID] = int(1 / (self.meanAbsVelocity[objectID] + 10) * 10000)

        # self.velocity[objectID][0] = np.median(self.velocity[objectID][2])
        # self.velocity[objectID][1] = np.median(self.velocity[objectID][3])

        # numOfFrames = 20
        # actualNumOfFrames = np.min([(len(self.centroidsMedian[objectID][2])), numOfFrames])
        # self.velocity[objectID][0] = np.round((self.centroidsMedian[objectID][2][-1] - self.centroidsMedian[objectID][2][-actualNumOfFrames]) / actualNumOfFrames, 1)
        # self.velocity[objectID][1] = np.round((self.centroidsMedian[objectID][3][-1] - self.centroidsMedian[objectID][3][-actualNumOfFrames]) / actualNumOfFrames, 1)

    def updateXYangle(self, objectID):

        x1 = self.centroidsMedian[objectID][2][0]
        y1 = self.centroidsMedian[objectID][3][0]
        x2 = self.centroidsMedian[objectID][2][-1]
        y2 = self.centroidsMedian[objectID][3][-1]

        deltaX = x2 - x1
        deltaY = y2 - y1
        absLength = np.sqrt(deltaX ** 2 + deltaY ** 2)

        if absLength > 0:
            xAng = int(np.degrees(np.arccos(deltaX / absLength)))
            yAng = int(np.degrees(np.arcsin(deltaY / absLength)))

            self.xAngle[objectID][1].append(xAng)
            self.yAngle[objectID][1].append(yAng)

            # taking the last value
            self.xAngle[objectID][0] = self.xAngle[objectID][1][-1]
            self.yAngle[objectID][0] = self.yAngle[objectID][1][-1]

    def updatePathAngle(self, objectID, previousCentroids):

        # calculating path angle on the last numOfFrames
        # numOfFrames = 40
        # actualNumOfFrames = np.min([(len(self.pathAngle[objectID][1])), numOfFrames])
        # x1 = self.centroidsMedian[objectID][2][-actualNumOfFrames]
        # y1 = self.centroidsMedian[objectID][3][-actualNumOfFrames]

        startCentroid = 0  # int(len(self.centroidsMedian[objectID][2]) / 2)

        x1 = self.centroidsMedian[objectID][2][startCentroid]
        y1 = self.centroidsMedian[objectID][3][startCentroid]
        x2 = self.centroidsMedian[objectID][2][-1]
        y2 = self.centroidsMedian[objectID][3][-1]

        angle = int(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 0:
            angle += 360

        self.pathAngle[objectID][1].append(angle)

        # taking the last path angle
        self.pathAngle[objectID][0] = self.pathAngle[objectID][1][-1]

    def rescale_BB_to_full_image(self, bboxStartX, bboxEndX, bboxStartY, bboxEndY, width, height):
        full_height, full_width, _ = self.full_frame.shape

        # Calculate the crop coordinates based on the known bbox coordinates
        crop_start_x = int(bboxStartX * full_width / width)
        crop_end_x = int(bboxEndX * full_width / width)
        crop_start_y = int(bboxStartY * full_height / height)
        crop_end_y = int(bboxEndY * full_height / height)

        cropped_image = self.full_frame[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        return cropped_image

    def crop_object(self, objectID, bbox, frame):
        # The optimal margin for cropping objects from images in a CNN classifier depends on several factors,
        # including the size of the objects, the complexity of the background, and the resolution of the images. Here
        # are some considerations to help you determine the margin:
        # 1. Object Size: If the objects you are detecting or classifying are relatively small compared to the image
        # size, you may need a larger margin to capture sufficient context around the object. This helps the CNN
        # understand the object's surroundings for better classification.
        # 2. Background Complexity: If the background of your images is cluttered or contains a lot
        # of irrelevant information, a larger margin can help the CNN focus on the object by providing more context
        # and reducing the impact of background noise.
        # 3. Resolution: Higher-resolution images can afford smaller margins because they contain more detail, allowing
        # the CNN to learn features effectively even with a tighter crop. Lower-resolution images may require larger
        # margins to compensate for the lack of detail.
        BB_margin = 3  # [pixels]
        bboxStartX = bbox[0] + BB_margin
        bboxStartY = bbox[1] + BB_margin
        bboxEndX = bbox[2] - BB_margin
        bboxEndY = bbox[3] - BB_margin

        if self.use_unscaled_frame_for_inference:
            cropped_image = self.rescale_BB_to_full_image(bboxStartX, bboxEndX, bboxStartY,
                                                          bboxEndY, self.width, self.height)
        else:
            cropped_image = frame[bboxStartY:bboxEndY, bboxStartX:bboxEndX]

        # cv.imshow('Cropped Event', cropped_image)

        cropped_height, cropped_width, _ = cropped_image.shape

        cropped_image = cv2.resize(cropped_image, self.image_resize, interpolation=cv2.INTER_NEAREST)

        # Documenting only images in the allowed size
        if self.min_dim <= cropped_width <= self.max_dim and self.min_dim <= cropped_height <= self.max_dim:
            # Resizing the cropped images to a constant size is crucial in terms of memory allocation in the list of
            # cropped images. Without the resizing to a constant size, the memory managment allocates large memory
            # portion for the cropped images list. The resize is done to the required size of the train and the
            # inference of the classifier, preventing data loss.
            # More details in - https://www.opensourceforu.com/2021/05/memory-management-in-lists-and-tuples/
            self.cropped_images[objectID].append(cropped_image)

        return cropped_image

    def build_xgboost_inference_row(self, objectID):
        if self.pathLength_path_history[objectID]:  # check if we have data in path history
            # The order of the built row should be the same as in xgboost_train.py ('train' function)
            X = [self.vX_path_history[objectID][-1],
                 self.vY_path_history[objectID][-1],
                 self.stdX_path_history[objectID][-1],
                 self.stdY_path_history[objectID][-1],
                 self.pathLength_path_history[objectID][-1],
                 self.aspectRatio_path_history[objectID][-1],
                 self.velocitySTD_path_history[objectID][-1],
                 self.lifeTime_path_history[objectID][-1],
                 self.sqrt_area_path_history[objectID][-1]
                 ]
        else:
            X = None

        return X

    def updateLabel(self, objectID, predicted_class_label, predicted_probability, predictions_probabilities, features):

        if not self.majorityVoting:
            self.classProbMeanCounter[objectID] += 1
            n = self.classProbMeanCounter[objectID]
            if n == 1:
                self.classesProbabilities[objectID] = predictions_probabilities
            else:
                self.classesProbabilities[objectID] = (self.classesProbabilities[objectID] * (
                        n - 1) + predictions_probabilities) / n

            precision = 3  # using round gives wierd result. So we use this method
            self.classesProbabilities[objectID] = np.array(
                [int(number * 10 ** precision) / (10 ** precision) for number in
                 self.classesProbabilities[objectID]])

            predicted_class_index = np.argmax(self.classesProbabilities[objectID])
            predicted_class_label = self.trt.labels.get(predicted_class_index, "Unknown")
            predicted_probability = self.classesProbabilities[objectID][predicted_class_index]

            if predicted_probability < self.classConfidence:
                predicted_class_label = 'Unknown'
                predicted_probability = np.nan

            self.label[objectID][0] = predicted_class_label
            self.labelProbability[objectID][0] = predicted_probability

        else:  # Use majority voting

            if predicted_probability < self.classConfidence:
                predicted_class_label = 'Unknown'
                predicted_probability = np.nan

            self.label[objectID][1].append(predicted_class_label)
            self.labelProbability[objectID][1].append(predicted_probability)

            majority_label = mode(self.label[objectID][1])  # Find the majority vote

            self.label[objectID][0] = majority_label
            # self.label[objectID][0] = predicted_class_label # For debug - show the current label (and not the majority voting)

            labels_array = np.array(self.label[objectID][1])
            probabilities_array = np.array(self.labelProbability[objectID][1])
            self.labelProbability[objectID][0] = calculate_mean_probability(labels_array, probabilities_array,
                                                                            majority_label)

            if features is not None:
                # start = time.time()
                self.object_features[objectID][1].append(features)
                features_array = np.array(
                    self.object_features[objectID][1])  # Convert the list of arrays to a NumPy array
                self.object_features[objectID][0] = np.mean(features_array,
                                                            axis=0)  # Calculate the column-wise averages
                # print(f'{round((time.time()-start)*1000,1)}')

        # end = time.time()
        # print(f'{inference_time}msec {predicted_class_label} ({round((end - start) * 1000, 1)})')

    def update(self, rects, frame, frameID):
        self.events = []
        self.deletedEventsIDs = []
        self.newDetections = 0

        self.objectID_path_history_final = []
        self.bboxPath = {}
        self.meanAbsVelocity_path_history_final = {}
        self.vX_path_history_final = {}
        self.vY_path_history_final = {}
        self.velocitySTD_path_history_final = {}
        self.aspectRatio_path_history_final = {}
        self.stdX_path_history_final = {}
        self.stdY_path_history_final = {}
        self.absStd_path_history_final = {}
        self.lifeTime_path_history_final = {}
        self.sqrt_area_path_history_final = {}
        self.pathLength_path_history_final = {}

        images_inference_list = []
        xgboost_inference_list = []
        inference_object_ID = []

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappearedTime.keys()):
                self.disappearedTime[objectID] = time.time() - self.startDisappearedTime[objectID]
                self.lifeTime[objectID] = time.time() - self.startTime[objectID]

                # We don't update median area and aspect ratio, centroids and velocity

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappearedTime[objectID] > self.maxDisappearedTime:
                    self.lifeTime[objectID] -= self.maxDisappearedTime
                    self.deregister(objectID, frameID)

                # if object appeared only one time, and in next frame it was not seen, we treet it as noise and
                #  deregister it.
                elif self.lifeTime[objectID] < self.minVisibleTime:
                    self.deregister(objectID, frameID)

            # return early as there are no centroids or tracking info
            # to update
            return self.centroids, self.lifeTime, self.area, self.boundingBoxs, self.centroidsMedian, \
                self.velocity, self.meanAbsVelocity, self.velocity_m1, self.velocitySTD, self.aspectRatio, self.deletedEventsIDs, \
                self.events, self.newDetections, self.sensorID, self.absStd, self.std, self.pathLength, \
                self.xAngle, self.yAngle, self.pathAngle, self.R, self.G, self.B, self.label, self.labelProbability, \
                self.startFrame, \
                self.objectID_path_history_final, \
                self.bboxPath, \
                self.meanAbsVelocity_path_history_final, \
                self.vX_path_history_final, \
                self.vY_path_history_final, \
                self.velocitySTD_path_history_final, \
                self.aspectRatio_path_history_final, \
                self.stdX_path_history_final, \
                self.stdY_path_history_final, \
                self.absStd_path_history_final, \
                self.lifeTime_path_history_final, \
                self.sqrt_area_path_history_final, \
                self.pathLength_path_history_final

        bboxs = np.zeros((len(rects), 4), dtype="int")
        centroids = np.zeros((len(rects), 2), dtype="int")
        sensorIDs = np.zeros((len(rects), 1), dtype="int")
        areas = np.zeros((len(rects), 1), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY, sensorID, area)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            bboxs[i] = (startX, startY, endX, endY)
            centroids[i] = (cX, cY)
            sensorIDs[i] = sensorID
            areas[i] = int(np.sqrt(area))

        if len(self.centroids) == 0:
            if len(centroids) <= self.maxNewDetections:
                for i in range(0, len(centroids)):
                    self.register(centroids[i], bboxs[i], frameID, sensorIDs[i], areas[i][0])
            else:
                self.newDetections = -1
        # otherwise, are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.centroids.keys())
            existing_object_centroids = list(self.centroids.values())

            existing_bounding_boxes = np.array([self.boundingBoxs[objectID][-1][:4] for objectID in objectIDs])

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            # D = dist.cdist(existing_object_centroids, centroids)
            ## D = diobjectCentroidsst.cdist(objectPredictedCentroids, centroids)

            # Compute combined metric
            # alpha = 0 --> score is only IoU
            # alpha = 1 --> score is only Euclidean distance (max distance in normalized to 1)
            # alpha = 0.5 --> score is blend half from each method
            euclidean_distances, D = combined_metric(existing_object_centroids, centroids, existing_bounding_boxes, bboxs, alpha=self.IoU_dist_alpha_blend)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # print(D[row, col])
                if euclidean_distances[row, col] > self.maxDistance:
                    # print(f'skip update do to max distance {objectIDs[row]}')
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.startDisappearedTime[objectID] = time.time()

                self.lifeTime[objectID] = time.time() - self.startTime[objectID]

                self.boundingBoxs[objectID].append(np.append(bboxs[col], frameID).tolist())

                previousCentroids = self.centroids[objectID]
                self.centroids[objectID] = centroids[col]

                if self.updateVelocityFlag:
                    self.updateVelocity(objectID, previousCentroids)

                if self.updateAreaAspectRatioFlag:
                    self.updateAreaAspectRatioMedian(objectID, areas[col][0])

                # The order of the update is important. Check before changing update order.
                self.updateCentroidsMedian(objectID)

                if self.updatePathAngleFlag:
                    self.updatePathAngle(objectID, previousCentroids)

                if self.updatexyAngleFlag:
                    self.updateXYangle(objectID)

                cropped_object = self.crop_object(objectID, bboxs[col], frame)

                if self.updateRGBFlag:
                    self.updateRGB(objectID, cropped_object)

                if self.imageClassificationFlag:
                    if (self.lifeTime[objectID] >= self.minTimeToStartInference) and (
                            (frameID - self.startFrame[objectID]) % self.inferenceFrequency == 0) or \
                            (self.label[objectID][0] == ''):

                        # The performance of the inference frequency is stochastic to some extent. The determination of the 'perform
                        # inference' flag is also influenced by the starting frame of the event. Therefore, if all or a large number
                        # of events start in the same frame, it could reduce performance when all these events are present. However,
                        # on average, each event starts with at equal random distribution frame, resulting in an overall improvement
                        # in performance.

                        temp_xgboost_row = self.build_xgboost_inference_row(objectID)
                        if temp_xgboost_row is not None:
                            xgboost_inference_list.append(temp_xgboost_row)
                            inference_object_ID.append(objectID)
                            images_inference_list.append(cropped_object)

                # Update features history (using later for XGboost modality training)
                self.meanAbsVelocity_path_history[objectID].append(self.meanAbsVelocity[objectID])
                self.vX_path_history[objectID].append(self.velocity[objectID][0])
                self.vY_path_history[objectID].append(self.velocity[objectID][1])
                self.velocitySTD_path_history[objectID].append(self.velocitySTD[objectID])
                self.aspectRatio_path_history[objectID].append(self.aspectRatio[objectID][1])
                self.stdX_path_history[objectID].append(self.std[objectID][0])
                self.stdY_path_history[objectID].append(self.std[objectID][1])
                self.absStd_path_history[objectID].append(self.absStd[objectID])
                self.lifeTime_path_history[objectID].append(np.round(self.lifeTime[objectID], 1))
                self.sqrt_area_path_history[objectID].append(self.area[objectID][1])
                self.pathLength_path_history[objectID].append(self.pathLength[objectID])

                # deregister event if it's length is bigger then the buffer clip length
                if self.lifeTime[objectID] >= self.maxClipLengthSec:
                    self.deregister(objectID, frameID)

                # Reset event label if it's background to allow background event to become other event if needed.
                elif self.label[objectID][0] == 'Background':
                    self.label[objectID] = ['Background', []]  # [majority voting label, [accumulated labels]]
                    self.labelProbability[objectID] = [self.labelProbability[objectID][0],
                                                       []]  # [mean majority voting probability, [accumulated probabilities]]

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            if xgboost_inference_list:
                startTime = time.time()
                if self.use_xgboost_classification_modality:
                    motion_classes, _ = self.xgboost.infer(xgboost_inference_list)
                else:
                    motion_classes = np.ones(len(xgboost_inference_list))
                motion_inference_time = round((time.time() - startTime) * 1000, 1)
                # print(f'XGBoost motion inference time: {motion_inference_time}msec')
                # For faster inference might need to use other dedicated library for real time inference: https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35
                # Other option for fast inference might be using other similar libraries like lightBGM, or catBoost.
                # LightBGM was developed after XGBoost, and it addresses weakness points of XGBoost.

                trt_inference_time = 0
                for Ob_Id, motion_predicted_class, cropped_object_for_infer in zip(inference_object_ID, motion_classes,
                                                                                   images_inference_list):
                    trtInfTime = 0
                    if self.startTime.get(Ob_Id) is not None:  # Check that the object ID was not deleted meanwhile
                        if motion_predicted_class == 0:
                            # XGboost predicted background
                            predicted_class_label = 'Background'
                            predicted_probability = 2  # marks that it was lables as background with the xgboost
                            predictions_probabilities = None
                            features = None
                        else:  # we perform CNN inference only if xgboost predict it's not background
                            startTime = time.time()
                            predicted_class_label, predicted_probability, inference_time, predictions_probabilities, \
                                features = self.trt.infer(cropped_object_for_infer)
                            trtInfTime = round((time.time() - startTime) * 1000, 1)

                        self.updateLabel(Ob_Id, predicted_class_label, predicted_probability, predictions_probabilities,
                                         features)

                    trt_inference_time += trtInfTime

                # if xgboost_inference_list:
                #     print(
                #         f'{len(motion_classes)} events, {motion_inference_time:.1f}msec xgboost, {trt_inference_time:.1f}msec trt')

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            # if D.shape[0] >= D.shape[1]:
            # loop over the unused row indexes
            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                self.disappearedTime[objectID] = time.time() - self.startDisappearedTime[objectID]
                self.lifeTime[objectID] = time.time() - self.startTime[objectID]

                # We don't update median area and aspect ratio, centroids and velocity

                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                if self.disappearedTime[objectID] > self.maxDisappearedTime:
                    self.lifeTime[objectID] -= self.maxDisappearedTime
                    self.deregister(objectID, frameID)

                elif self.lifeTime[objectID] < self.minVisibleTime:
                    self.deregister(objectID, frameID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            # else:
            if len(unusedCols) <= self.maxNewDetections:
                for col in unusedCols:
                    self.register(centroids[col], bboxs[col].tolist(), frameID, sensorIDs[col], areas[col][0])
            else:
                self.newDetections = -1

        return self.centroids, self.lifeTime, self.area, self.boundingBoxs, self.centroidsMedian, \
            self.velocity, self.meanAbsVelocity, self.velocity_m1, self.velocitySTD, self.aspectRatio, self.deletedEventsIDs, \
            self.events, self.newDetections, self.sensorID, self.absStd, self.std, \
            self.pathLength, self.xAngle, self.yAngle, self.pathAngle, self.R, self.G, self.B, self.label, self.labelProbability, \
            self.startFrame, \
            self.objectID_path_history_final, \
            self.bboxPath, \
            self.meanAbsVelocity_path_history_final, \
            self.vX_path_history_final, \
            self.vY_path_history_final, \
            self.velocitySTD_path_history_final, \
            self.aspectRatio_path_history_final, \
            self.stdX_path_history_final, \
            self.stdY_path_history_final, \
            self.absStd_path_history_final, \
            self.lifeTime_path_history_final, \
            self.sqrt_area_path_history_final, \
            self.pathLength_path_history_final
