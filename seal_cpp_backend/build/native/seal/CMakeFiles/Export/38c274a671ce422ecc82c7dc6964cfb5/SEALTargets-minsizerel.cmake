#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SEAL::seal" for configuration "MinSizeRel"
set_property(TARGET SEAL::seal APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SEAL::seal PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "C;CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/seal-4.1.lib"
  )

list(APPEND _cmake_import_check_targets SEAL::seal )
list(APPEND _cmake_import_check_files_for_SEAL::seal "${_IMPORT_PREFIX}/lib/seal-4.1.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
