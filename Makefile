# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 4.0

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\MinGW\bin\cmake.exe

# The command to remove a file.
RM = C:\MinGW\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\AI\GODOT\libs\ironui-diffusion.cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\AI\GODOT\libs\ironui-diffusion.cpp

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "No interactive CMake dialog available..."
	C:\MinGW\bin\cmake.exe -E echo "No interactive CMake dialog available."
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	C:\MinGW\bin\cmake.exe --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components
.PHONY : list_install_components/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Install the project..."
	C:\MinGW\bin\cmake.exe -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Install the project..."
	C:\MinGW\bin\cmake.exe -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Installing only the local directory..."
	C:\MinGW\bin\cmake.exe -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Installing only the local directory..."
	C:\MinGW\bin\cmake.exe -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Installing the project stripped..."
	C:\MinGW\bin\cmake.exe -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Installing the project stripped..."
	C:\MinGW\bin\cmake.exe -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start D:\AI\GODOT\libs\ironui-diffusion.cpp\CMakeFiles D:\AI\GODOT\libs\ironui-diffusion.cpp\\CMakeFiles\progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start D:\AI\GODOT\libs\ironui-diffusion.cpp\CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles\Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named stable-diffusion

# Build rule for target.
stable-diffusion: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 stable-diffusion
.PHONY : stable-diffusion

# fast build rule for target.
stable-diffusion/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/build
.PHONY : stable-diffusion/fast

#=============================================================================
# Target rules for targets named ggml-base

# Build rule for target.
ggml-base: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 ggml-base
.PHONY : ggml-base

# fast build rule for target.
ggml-base/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\CMakeFiles\ggml-base.dir\build.make ggml/src/CMakeFiles/ggml-base.dir/build
.PHONY : ggml-base/fast

#=============================================================================
# Target rules for targets named ggml

# Build rule for target.
ggml: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 ggml
.PHONY : ggml

# fast build rule for target.
ggml/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\CMakeFiles\ggml.dir\build.make ggml/src/CMakeFiles/ggml.dir/build
.PHONY : ggml/fast

#=============================================================================
# Target rules for targets named ggml-cpu

# Build rule for target.
ggml-cpu: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 ggml-cpu
.PHONY : ggml-cpu

# fast build rule for target.
ggml-cpu/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\CMakeFiles\ggml-cpu.dir\build.make ggml/src/CMakeFiles/ggml-cpu.dir/build
.PHONY : ggml-cpu/fast

#=============================================================================
# Target rules for targets named ggml-vulkan

# Build rule for target.
ggml-vulkan: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 ggml-vulkan
.PHONY : ggml-vulkan

# fast build rule for target.
ggml-vulkan/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\ggml-vulkan\CMakeFiles\ggml-vulkan.dir\build.make ggml/src/ggml-vulkan/CMakeFiles/ggml-vulkan.dir/build
.PHONY : ggml-vulkan/fast

#=============================================================================
# Target rules for targets named vulkan-shaders-gen

# Build rule for target.
vulkan-shaders-gen: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 vulkan-shaders-gen
.PHONY : vulkan-shaders-gen

# fast build rule for target.
vulkan-shaders-gen/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\ggml-vulkan\CMakeFiles\vulkan-shaders-gen.dir\build.make ggml/src/ggml-vulkan/CMakeFiles/vulkan-shaders-gen.dir/build
.PHONY : vulkan-shaders-gen/fast

#=============================================================================
# Target rules for targets named vulkan-shaders-gen-build

# Build rule for target.
vulkan-shaders-gen-build: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 vulkan-shaders-gen-build
.PHONY : vulkan-shaders-gen-build

# fast build rule for target.
vulkan-shaders-gen-build/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\ggml-vulkan\CMakeFiles\vulkan-shaders-gen-build.dir\build.make ggml/src/ggml-vulkan/CMakeFiles/vulkan-shaders-gen-build.dir/build
.PHONY : vulkan-shaders-gen-build/fast

#=============================================================================
# Target rules for targets named vulkan-shaders-gen-install

# Build rule for target.
vulkan-shaders-gen-install: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 vulkan-shaders-gen-install
.PHONY : vulkan-shaders-gen-install

# fast build rule for target.
vulkan-shaders-gen-install/fast:
	$(MAKE) $(MAKESILENT) -f ggml\src\ggml-vulkan\CMakeFiles\vulkan-shaders-gen-install.dir\build.make ggml/src/ggml-vulkan/CMakeFiles/vulkan-shaders-gen-install.dir/build
.PHONY : vulkan-shaders-gen-install/fast

#=============================================================================
# Target rules for targets named zip

# Build rule for target.
zip: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 zip
.PHONY : zip

# fast build rule for target.
zip/fast:
	$(MAKE) $(MAKESILENT) -f thirdparty\CMakeFiles\zip.dir\build.make thirdparty/CMakeFiles/zip.dir/build
.PHONY : zip/fast

#=============================================================================
# Target rules for targets named shared

# Build rule for target.
shared: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 shared
.PHONY : shared

# fast build rule for target.
shared/fast:
	$(MAKE) $(MAKESILENT) -f shared\CMakeFiles\shared.dir\build.make shared/CMakeFiles/shared.dir/build
.PHONY : shared/fast

model.obj: model.cpp.obj
.PHONY : model.obj

# target to build an object file
model.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/model.cpp.obj
.PHONY : model.cpp.obj

model.i: model.cpp.i
.PHONY : model.i

# target to preprocess a source file
model.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/model.cpp.i
.PHONY : model.cpp.i

model.s: model.cpp.s
.PHONY : model.s

# target to generate assembly for a file
model.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/model.cpp.s
.PHONY : model.cpp.s

stable-diffusion.obj: stable-diffusion.cpp.obj
.PHONY : stable-diffusion.obj

# target to build an object file
stable-diffusion.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.obj
.PHONY : stable-diffusion.cpp.obj

stable-diffusion.i: stable-diffusion.cpp.i
.PHONY : stable-diffusion.i

# target to preprocess a source file
stable-diffusion.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.i
.PHONY : stable-diffusion.cpp.i

stable-diffusion.s: stable-diffusion.cpp.s
.PHONY : stable-diffusion.s

# target to generate assembly for a file
stable-diffusion.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/stable-diffusion.cpp.s
.PHONY : stable-diffusion.cpp.s

upscaler.obj: upscaler.cpp.obj
.PHONY : upscaler.obj

# target to build an object file
upscaler.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/upscaler.cpp.obj
.PHONY : upscaler.cpp.obj

upscaler.i: upscaler.cpp.i
.PHONY : upscaler.i

# target to preprocess a source file
upscaler.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/upscaler.cpp.i
.PHONY : upscaler.cpp.i

upscaler.s: upscaler.cpp.s
.PHONY : upscaler.s

# target to generate assembly for a file
upscaler.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/upscaler.cpp.s
.PHONY : upscaler.cpp.s

util.obj: util.cpp.obj
.PHONY : util.obj

# target to build an object file
util.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/util.cpp.obj
.PHONY : util.cpp.obj

util.i: util.cpp.i
.PHONY : util.i

# target to preprocess a source file
util.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/util.cpp.i
.PHONY : util.cpp.i

util.s: util.cpp.s
.PHONY : util.s

# target to generate assembly for a file
util.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\stable-diffusion.dir\build.make CMakeFiles/stable-diffusion.dir/util.cpp.s
.PHONY : util.cpp.s

# Help Target
help:
	@echo The following are some of the valid targets for this Makefile:
	@echo ... all (the default if no target is provided)
	@echo ... clean
	@echo ... depend
	@echo ... edit_cache
	@echo ... install
	@echo ... install/local
	@echo ... install/strip
	@echo ... list_install_components
	@echo ... rebuild_cache
	@echo ... vulkan-shaders-gen
	@echo ... vulkan-shaders-gen-build
	@echo ... vulkan-shaders-gen-install
	@echo ... ggml
	@echo ... ggml-base
	@echo ... ggml-cpu
	@echo ... ggml-vulkan
	@echo ... shared
	@echo ... stable-diffusion
	@echo ... zip
	@echo ... model.obj
	@echo ... model.i
	@echo ... model.s
	@echo ... stable-diffusion.obj
	@echo ... stable-diffusion.i
	@echo ... stable-diffusion.s
	@echo ... upscaler.obj
	@echo ... upscaler.i
	@echo ... upscaler.s
	@echo ... util.obj
	@echo ... util.i
	@echo ... util.s
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles\Makefile.cmake 0
.PHONY : cmake_check_build_system

