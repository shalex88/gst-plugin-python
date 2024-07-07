############# Custom element configuration #################################
ELEMENT_NAME = 'preprocessing'

import cv2
from vidstab.VidStab import VidStab
stabilizer = VidStab(processing_max_dim=200)

format_to_channels = {
    "UYVY": (2, cv2.COLOR_YUV2BGR_UYVY, cv2.COLOR_BGR2YUV_UYVY),
    "BGR": (3, None, None),
    "RGB": (3, cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB),
    "YUY2": (2, cv2.COLOR_YUV2BGR_YUY2, cv2.COLOR_BGR2YUV_YUY2),
}

def custom_processing(frame, width, height, format):
    global stabilizer
    global format_to_channels

    num_of_channels = format_to_channels.get(format)[0]
    convert_to_bgr = format_to_channels.get(format)[1]
    convert_from_bgr = format_to_channels.get(format)[2]

    # Prepare frame for processing
    frame = frame.reshape(height, width, num_of_channels)
    if convert_to_bgr is not None:
        frame = cv2.cvtColor(frame, convert_to_bgr)

    # Apply stabilization
    frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=3)

    # Convert frame back to original format
    if convert_from_bgr is not None:
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
