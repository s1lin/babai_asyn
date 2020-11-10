# For some reason, CMake can't figure out PIC flag for Cray
if(CXX_COMPILER_CRAY)
  add_compile_options(-fPIC)
endif()

if(WITH_MC64)
  target_link_libraries(orderingscaling ${MC64_LIB} ${FORTLDFLAGS})
endif(WITH_MC64)


add_library(sils SHARED ./include/SILS.h ./source/SILS.cpp)
set_property(TARGET sils PROPERTY POSITION_INDEPENDENT_CODE ON)

# Tags

set_source_files_properties(tags PROPERTIES GENERATED true)
add_custom_target ( tags
    COMMAND ctags -R --c++-kinds=+pf --extra=+q . 
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
  )

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
    message(STATUS "Found ctags: ${CTAGS_PATH}")
else()
	message(STATUS "ctags not found! Tags will not be generated.")
endif(CTAGS_PATH)


