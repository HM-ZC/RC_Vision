# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/edlib_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/edlib_ws/build

# Include any dependencies generated for this target.
include edlib/CMakeFiles/my_app.dir/depend.make

# Include the progress variables for this target.
include edlib/CMakeFiles/my_app.dir/progress.make

# Include the compile flags for this target's objects.
include edlib/CMakeFiles/my_app.dir/flags.make

edlib/CMakeFiles/my_app.dir/src/my_app.cpp.o: edlib/CMakeFiles/my_app.dir/flags.make
edlib/CMakeFiles/my_app.dir/src/my_app.cpp.o: /root/edlib_ws/src/edlib/src/my_app.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/edlib_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object edlib/CMakeFiles/my_app.dir/src/my_app.cpp.o"
	cd /root/edlib_ws/build/edlib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_app.dir/src/my_app.cpp.o -c /root/edlib_ws/src/edlib/src/my_app.cpp

edlib/CMakeFiles/my_app.dir/src/my_app.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_app.dir/src/my_app.cpp.i"
	cd /root/edlib_ws/build/edlib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/edlib_ws/src/edlib/src/my_app.cpp > CMakeFiles/my_app.dir/src/my_app.cpp.i

edlib/CMakeFiles/my_app.dir/src/my_app.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_app.dir/src/my_app.cpp.s"
	cd /root/edlib_ws/build/edlib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/edlib_ws/src/edlib/src/my_app.cpp -o CMakeFiles/my_app.dir/src/my_app.cpp.s

# Object files for target my_app
my_app_OBJECTS = \
"CMakeFiles/my_app.dir/src/my_app.cpp.o"

# External object files for target my_app
my_app_EXTERNAL_OBJECTS =

/root/edlib_ws/devel/lib/edlib/my_app: edlib/CMakeFiles/my_app.dir/src/my_app.cpp.o
/root/edlib_ws/devel/lib/edlib/my_app: edlib/CMakeFiles/my_app.dir/build.make
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libimage_transport.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libmessage_filters.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libclass_loader.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libPocoFoundation.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libdl.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libroslib.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/librospack.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libpython3.8.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libcamera_info_manager.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libcamera_calibration_parsers.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libroscpp.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libpthread.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/librosconsole.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libxmlrpcpp.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libroscpp_serialization.so
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/librostime.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /opt/ros/noetic/lib/libcpp_common.so
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
/root/edlib_ws/devel/lib/edlib/my_app: edlib/CMakeFiles/my_app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/edlib_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /root/edlib_ws/devel/lib/edlib/my_app"
	cd /root/edlib_ws/build/edlib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
edlib/CMakeFiles/my_app.dir/build: /root/edlib_ws/devel/lib/edlib/my_app

.PHONY : edlib/CMakeFiles/my_app.dir/build

edlib/CMakeFiles/my_app.dir/clean:
	cd /root/edlib_ws/build/edlib && $(CMAKE_COMMAND) -P CMakeFiles/my_app.dir/cmake_clean.cmake
.PHONY : edlib/CMakeFiles/my_app.dir/clean

edlib/CMakeFiles/my_app.dir/depend:
	cd /root/edlib_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/edlib_ws/src /root/edlib_ws/src/edlib /root/edlib_ws/build /root/edlib_ws/build/edlib /root/edlib_ws/build/edlib/CMakeFiles/my_app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : edlib/CMakeFiles/my_app.dir/depend

