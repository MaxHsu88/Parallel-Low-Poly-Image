# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ipc21/ipc21s05/Parallel-Low-Poly-Image

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ipc21/ipc21s05/Parallel-Low-Poly-Image

# Include any dependencies generated for this target.
include source/CMakeFiles/ParaLowPoly.dir/depend.make

# Include the progress variables for this target.
include source/CMakeFiles/ParaLowPoly.dir/progress.make

# Include the compile flags for this target's objects.
include source/CMakeFiles/ParaLowPoly.dir/flags.make

source/CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.o: source/CMakeFiles/ParaLowPoly.dir/flags.make
source/CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.o: source/LowPolySolver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ipc21/ipc21s05/Parallel-Low-Poly-Image/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object source/CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.o"
	cd /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.o -c /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source/LowPolySolver.cpp

source/CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.i"
	cd /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source/LowPolySolver.cpp > CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.i

source/CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.s"
	cd /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source/LowPolySolver.cpp -o CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.s

# Object files for target ParaLowPoly
ParaLowPoly_OBJECTS = \
"CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.o"

# External object files for target ParaLowPoly
ParaLowPoly_EXTERNAL_OBJECTS =

ParaLowPoly: source/CMakeFiles/ParaLowPoly.dir/LowPolySolver.cpp.o
ParaLowPoly: source/CMakeFiles/ParaLowPoly.dir/build.make
ParaLowPoly: /opt/cuda/lib64/libcudart_static.a
ParaLowPoly: /usr/lib/librt.so
ParaLowPoly: /home/ipc21/ipc21s05/opencv/lib/libopencv_imgcodecs.so.3.4.3
ParaLowPoly: /home/ipc21/ipc21s05/opencv/lib/libopencv_imgproc.so.3.4.3
ParaLowPoly: /home/ipc21/ipc21s05/opencv/lib/libopencv_core.so.3.4.3
ParaLowPoly: source/CMakeFiles/ParaLowPoly.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ipc21/ipc21s05/Parallel-Low-Poly-Image/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../ParaLowPoly"
	cd /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ParaLowPoly.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
source/CMakeFiles/ParaLowPoly.dir/build: ParaLowPoly

.PHONY : source/CMakeFiles/ParaLowPoly.dir/build

source/CMakeFiles/ParaLowPoly.dir/clean:
	cd /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source && $(CMAKE_COMMAND) -P CMakeFiles/ParaLowPoly.dir/cmake_clean.cmake
.PHONY : source/CMakeFiles/ParaLowPoly.dir/clean

source/CMakeFiles/ParaLowPoly.dir/depend:
	cd /home/ipc21/ipc21s05/Parallel-Low-Poly-Image && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ipc21/ipc21s05/Parallel-Low-Poly-Image /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source /home/ipc21/ipc21s05/Parallel-Low-Poly-Image /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source /home/ipc21/ipc21s05/Parallel-Low-Poly-Image/source/CMakeFiles/ParaLowPoly.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : source/CMakeFiles/ParaLowPoly.dir/depend

