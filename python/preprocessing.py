############# Custom element configuration #################################
ELEMENT_NAME = 'preprocessing'

def custom_processing(frame):
    # Example: Invert frame
    frame[:] = 255 - frame[:]
    # ariel
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
        return True

    def do_transform_ip(self, inbuf: Gst.Buffer) -> Gst.FlowReturn:
        try:
            success, map_info = inbuf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

            original_frame = np.frombuffer(map_info.data, dtype=np.uint8)
            processed_frame = self.custom_processing_func(original_frame)
            np.copyto(original_frame, processed_frame)
            inbuf.unmap(map_info)

            return Gst.FlowReturn.OK
        except Exception as e:
            print(f"Error occurred during custom processing: {e}")
            return Gst.FlowReturn.ERROR

GObject.type_register(CustomTransformElement)
__gstelementfactory__ = (ELEMENT_NAME, Gst.Rank.NONE, CustomTransformElement)
