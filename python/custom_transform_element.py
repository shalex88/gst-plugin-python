############# Custom element configuration #################################
ELEMENT_NAME = 'custom_transform_element'
WIDTH = 640
HEIGHT = 480
CHANNELS = 3
FRAMERATE_NUM = 25
FRAMERATE_DENOM = 1
FORMAT = 'RGB'

# Custom processing can be added here
def custom_processing(frame):
    # Example: No processing
    processed_frame = frame

    # Example: Convert frame to grayscale
    # grayscale_frame = np.mean(frame, axis=2).astype(np.uint8)

    # Example: Convert frame to grayscale with OpenCV   
    # import cv2
    # grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # processed_frame = np.stack((grayscale_frame,)*3, axis=-1)

    return processed_frame
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

OCAPS = Gst.Caps.from_string('video/x-raw, format=(string){}, width=(int){}, height=(int){}, framerate=(fraction){}/{}'.format(FORMAT, WIDTH, HEIGHT, FRAMERATE_NUM, FRAMERATE_DENOM))
ICAPS = Gst.Caps.from_string('video/x-raw, format=(string){}, width=(int){}, height=(int){}, framerate=(fraction){}/{}'.format(FORMAT, WIDTH, HEIGHT, FRAMERATE_NUM, FRAMERATE_DENOM))

class CustomTransformElement(GstBase.BaseTransform):
    __gstmetadata__ = ('Custom Transform Element', 'Transform', 'Base for Custom Transform Element', 'Alex Sh')

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            OCAPS
        ),
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            ICAPS
        )
    )

    def __init__(self):
        GstBase.BaseTransform.__init__(self)

        self.element_name = ELEMENT_NAME
        self.custom_processing_func = custom_processing
        self.width = WIDTH
        self.height = HEIGHT
        self.channels = CHANNELS

    def do_transform_caps(self, direction, caps, filter_):
        if direction == Gst.PadDirection.SRC:
            res = ICAPS
        else:
            res = OCAPS

        if filter_:
            res = res.intersect(filter_)

        return res
    
    def do_fixate_caps(self, direction, caps, othercaps):
        if direction == Gst.PadDirection.SRC:
            return othercaps.fixate()
        else:
            so = othercaps.get_structure(0).copy()
            so.fixate_field_nearest_fraction("framerate",
                                             FRAMERATE_NUM,
                                             FRAMERATE_DENOM)
            so.fixate_field_nearest_int("width", WIDTH)
            so.fixate_field_nearest_int("height", HEIGHT)
            ret = Gst.Caps.new_empty()
            ret.append_structure(so)
            return ret.fixate()

    def do_transform_ip(self, inbuf: Gst.Buffer) -> Gst.FlowReturn:
        try:
            success, map_info = inbuf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

            frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame_data.reshape((self.height, self.width, self.channels))
            processed_frame = self.custom_processing_func(frame)
            np.copyto(frame_data, processed_frame.flatten())
            inbuf.unmap(map_info)

            return Gst.FlowReturn.OK
        except Exception as e:
            print(f"Error occurred during custom processing: {e}")
            return Gst.FlowReturn.ERROR

GObject.type_register(CustomTransformElement)
__gstelementfactory__ = (ELEMENT_NAME, Gst.Rank.NONE, CustomTransformElement)
