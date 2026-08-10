#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "roomie::roomie_pipeline" for configuration ""
set_property(TARGET roomie::roomie_pipeline APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(roomie::roomie_pipeline PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libroomie_pipeline.so"
  IMPORTED_SONAME_NOCONFIG "libroomie_pipeline.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS roomie::roomie_pipeline )
list(APPEND _IMPORT_CHECK_FILES_FOR_roomie::roomie_pipeline "${_IMPORT_PREFIX}/lib/libroomie_pipeline.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
