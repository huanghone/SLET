CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

SET(PROJECTNAME SLOT)
PROJECT(${PROJECTNAME})
set(CMAKE_CXX_STANDARD 17)
#####################################
# cmake settings
#####################################

#SET(CMAKE_VERBOSE_MAKEFILE ON)
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR}/)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")  
set(CMAKE_RELEASE_POSTFIX "" CACHE STRING "add a postfix, usually empty on windows")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(WIN32)
	SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
endif()

SET(CMAKE_INCLUDE_CURRENT_DIR ON)
#SET(CMAKE_AUTOMOC ON)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

#####################################
# compiler flags
#####################################
if(WIN32)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MD")

	set (COMPILE_FLAG
	"${CMAKE_CXX_FLAGS} -DNOMINMAX -DCVX_NO_GUI -DCVX_NO_CODEC /wd4996 /wd4267 /wd4244 /MP")
	# 4996 deprecated crt functions
	# 4267 size to int
	# 4244 float to int

	set(CMAKE_CXX_FLAGS ${COMPILE_FLAG})
	set(CMAKE_C_FLAGS ${COMPILE_FLAG})
else()
	SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
	SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
	#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -c -O3")
	#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MD")
endif()

#####################################
# 3rd libs
#####################################
if(WIN32)
	#set(QT_DIR D:/lib/qt/5.13.0/msvc2017_64)
	#set(CMAKE_PREFIX_PATH ${qtdir})

	set(OpenCV_DIR d:/lib/opencv4)
	SET(OpenCV_STATIC OFF)

	set(ASSIMP_DIR D:/lib/assimp)

	set(gflags_DIR d:/lib/gflags/lib/cmake/gflags)
	set(glog_DIR d:/lib/glog/lib/cmake/glog)
else()

endif()

find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(assimp REQUIRED)
find_package(glog REQUIRED)
find_package(OpenGL REQUIRED)

if(WIN32)
	SET(DEP_LIBS
		glog::glog 
		Qt5::Widgets 
		Qt5::OpenGL
		${OpenCV_LIBS}
		${OPENGL_LIBRARIES}
		${ASSIMP_LIBRARIES}
		zlibstatic.lib
	)
	message("assimp: ${ASSIMP_LIBRARIES}")
else()
	SET(DEP_LIBS
		glog::glog 
		Qt5::Widgets 
		Qt5::OpenGL
		${OpenCV_LIBS}
		${OPENGL_LIBRARIES}
		${ASSIMP_LIBRARIES}
	)
endif()

INCLUDE_DIRECTORIES(
	src
	${OpenCV_INCLUDE_DIRS} 
	${OPENGL_INCLUDE_DIR}
	${ASSIMP_INCLUDE_DIR}
)

link_directories(${PROJECT_ROOT}/lib)

#####################################
# here we go
#####################################

add_subdirectory(src/base)
add_subdirectory(src/mbt)
add_subdirectory(src/app)

#====================================================
# tracker
#====================================================

#ADD_EXECUTABLE(tracker app/tracker.cc)
#TARGET_LINK_LIBRARIES(tracker base mbt ${DEP_LIBS} zlibstatic.lib)
#set_target_properties(tracker PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_ROOT})
