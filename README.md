# gst-plugin-python

[Tutorial](https://medium.com/@jasonlife/writing-gstreamer-plugin-with-python-b98627cd24c1)

## Install

```bash
sudo apt install python3-gst-1.0 gstreamer1.0-python3-plugin-loader
# Test
gst-inspect-1.0 python
```

## Run

Current directory structure is mandatory
Run the following commands in the root directory while the plugins are in the 'python' directory

```bash
# Setup
export GST_PLUGIN_PATH=./

# Test
gst-inspect-1.0 custom_src_element
# Run
gst-launch-1.0 custom_src_element ! autovideosink

# Install
pip install numpy
# Test
gst-inspect-1.0 custom_transform_element
# Run
gst-launch-1.0 videotestsrc ! custom_transform_element ! autovideoconvert ! autovideosink -vvv
gst-launch-1.0 videotestsrc ! video/x-raw,width=1280,height=1080,framerate=60/1,format=RGB ! custom_transform_element ! autovideoconvert ! autovideosink -vvv
gst-launch-1.0 videotestsrc ! video/x-raw,width=640,height=480,framerate=30/1,format=YUY2 ! textoverlay text="Camera1" ! custom_transform_element ! nvvidconv ! autovideosink -vvv
```
