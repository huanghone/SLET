FILE(GLOB mbt_SRC *.hh *.cc *.cpp *.h *.hpp)
source_group("" FILES ${mbt_SRC})

FILE(GLOB glsl_SOURCES shader/*.glsl)
source_group("shader" FILES ${glsl_SOURCES})

add_library(mbt ${mbt_SRC} ${glsl_SOURCES})
target_link_libraries(mbt ${DEP_LIBS})