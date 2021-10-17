
file(GLOB base_SRC *.hh base/*.h *.cc)
source_group("" FILES ${base_SRC})

add_library(base ${base_SRC})
target_link_libraries(base ${DEP_LIBS})