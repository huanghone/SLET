set(run_on_video_SRC
	run_on_video.cc
)
source_group("" FILES ${run_on_video_SRC})
add_executable(run_on_video ${run_on_video_SRC})
target_link_libraries(run_on_video ${DEP_LIBS} base mbt)

set(TRACKER_TARGET
	run_on_video
)
set_target_properties(${TRACKER_TARGET} PROPERTIES FOLDER "tracker")
set_target_properties(${TRACKER_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_ROOT})