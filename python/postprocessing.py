############# Custom element configuration #################################
ELEMENT_NAME = 'postprocessing'

import cv2
import numpy as np
from centroidtracker import CentroidTracker

frameID = 0
frame_shape = (1080, 1920, 2)
mask_shape = (1080, 1920, 1)

npMat1 = np.random.randint(0, 255, size=frame_shape).astype(np.uint8)
cuMat = cv2.cuda_GpuMat(npMat1)

npMat2 = np.random.randint(0, 255, size=mask_shape).astype(np.uint8)
cuMatFg = cv2.cuda_GpuMat(npMat2)

fgbg = cv2.cuda.createBackgroundSubtractorMOG2(history=500, detectShadows=False, varThreshold=32)

format_to_channels = {
    "UYVY": (2, cv2.COLOR_YUV2BGR_UYVY, cv2.COLOR_BGR2YUV_UYVY),
    "BGR": (3, None, None),
    "RGB": (3, cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB),
    "YUY2": (2, cv2.COLOR_YUV2BGR_YUY2, cv2.COLOR_BGR2YUV_YUY2),
}

ct = CentroidTracker(nextObjectID=0, maxDisappearedTime=1, maxDistance=40,
                     minLifeTime=0,
                     maxNewDetections=15,
                     tensorrt_model=None,
                     classConfidence=0.5, minTimeToStartInference=0.5,
                     inferenceFrequency=3,
                     maxClipLengthMin=0.75, majorityVoting=False,
                     use_unscaled_frame_for_inference=False, width=frame_shape[1],
                     height=frame_shape[0],
                     use_xgboost_classification_modality=False,
                     xgboost=None,
                     IoU_dist_alpha_blend=0
                     )

def custom_processing(frame, width, height, format):
    global test_var, npMat1, fgbg, frame_shape, cuMat, cuMatFg

    global format_to_channels, ct, frameID

    frameID += 1

    num_of_channels = format_to_channels.get(format)[0]
    convert_to_bgr = format_to_channels.get(format)[1]
    convert_from_bgr = format_to_channels.get(format)[2]

    frame = frame.reshape(height, width, num_of_channels)

    frame = cv2.cvtColor(frame, convert_to_bgr)

    if True:

        frameID += 1

        ct.full_frame = frame.copy()

        cuMat.upload(frame)

        stream = cv2.cuda_Stream()
        fgbg.apply(cuMat, -1, stream, cuMatFg)

        fgmask = cuMatFg.download()

        fgmask = cv2.erode(fgmask, np.ones((3, 3), np.uint8), iterations=1)  # Remove initial noise from the frame
        fgmask = cv2.dilate(fgmask, np.ones((21, 21), np.uint8),
                            iterations=1)  # connect close pixel together to get full objects

        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(fgmask, 8, cv2.CV_32S)

        rects = []

        for i in range(1, numLabels):
            startX = stats[i, cv2.CC_STAT_LEFT]
            startY = stats[i, cv2.CC_STAT_TOP]
            endX = startX + stats[i, cv2.CC_STAT_WIDTH]
            endY = startY + stats[i, cv2.CC_STAT_HEIGHT]
            tempArea = stats[i, cv2.CC_STAT_AREA]
            sensorID = 0

            box = np.array([startX, startY, endX, endY, sensorID, tempArea])

            rects.append(box)

        objects, lifeTime, area, boundingBoxs, centroidsMean, velocity, meanAbsVelocity, velocity_m1, velocitySTD, \
            aspectRatio, deletedEventsIDs, events, newEventsCount, sensors, absStd, Std, pathLength, \
            xAngle, yAngle, pathAngle, R, G, B, predictedLabels, predictedLabelsProbability, \
            startFrame, \
            objectID_path_history_final, \
            bboxPath, \
            meanAbsVelocity_path_history_final, \
            vX_path_history_final, \
            vY_path_history_final, \
            velocitySTD_path_history_final, \
            aspectRatio_path_history_final, \
            stdX_path_history_final, \
            stdY_path_history_final, \
            absStd_path_history_final, \
            lifeTime_path_history_final, \
            sqrt_area_path_history_final, \
            pathLength_path_history_final = ct.update(rects,
                                                      frame,
                                                      frameID)

        for (objectID, bboxList) in boundingBoxs.items():
            bbox = bboxList[-1]
            lineWidth = 2
            margin = 5

            bboxStartX = int(bbox[0]) - margin
            bboxStartY = int(bbox[1]) - margin
            bboxEndX = int(bbox[2]) + margin
            bboxEndY = int(bbox[3]) + margin

            cv2.rectangle(frame, (bboxStartX, bboxStartY), (bboxEndX, bboxEndY), (0, 255, 0), lineWidth)
            cv2.putText(frame, f'ID {objectID}', (bboxStartX - 2, bboxStartY - 4 - margin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, convert_from_bgr)
    frame = frame.reshape(-1)
    return frame
############################################################################
import sys
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')

from gi.repository import Gst, GObject, GstBase
import numpy as np

print('Python version: {}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
print('NumPy version: {}'.format(np.__version__))

Gst.init(None)

class CustomTransformElement(GstBase.BaseTransform):
    __gstmetadata__ = ('Custom Transform Element', 'Transform', 'Base for Custom Transform Element', 'Alex Sh')

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string('video/x-raw')
        ),
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string('video/x-raw')
        )
    )

    def __init__(self):
        GstBase.BaseTransform.__init__(self)

        self.element_name = ELEMENT_NAME
        self.custom_processing_func = custom_processing

    def do_transform_caps(self, direction, caps, filter_):
        # Pass input caps to output caps directly
        return caps

    def do_set_caps(self, incaps, outcaps):
        self.incaps = incaps
        try:
            structure = self.incaps.get_structure(0)
            self.width = structure.get_value("width")
            self.height = structure.get_value("height")
            self.format = structure.get_value("format")
        except Exception as e:
            print(f"Error occurred during custom processing: {e}")
            return Gst.FlowReturn.ERROR
        return True

    def do_transform_ip(self, inbuf: Gst.Buffer) -> Gst.FlowReturn:
        try:
            success, map_info = inbuf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

            original_frame = np.frombuffer(map_info.data, dtype=np.uint8)
            processed_frame = self.custom_processing_func(original_frame, self.width, self.height, self.format)
            np.copyto(original_frame, processed_frame)
            inbuf.unmap(map_info)

            return Gst.FlowReturn.OK
        except Exception as e:
            print(f"Error occurred during custom processing: {e}")
            return Gst.FlowReturn.ERROR

GObject.type_register(CustomTransformElement)
__gstelementfactory__ = (ELEMENT_NAME, Gst.Rank.NONE, CustomTransformElement)
