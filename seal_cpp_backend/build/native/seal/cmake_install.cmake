# Install script for directory: C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/external/SEAL

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/BlindGEN_Server")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SEAL-4.1/seal/util" TYPE FILE FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/native/src/seal/util/config.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/lib/Debug/seal-4.1.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/lib/Release/seal-4.1.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/lib/MinSizeRel/seal-4.1.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/lib/RelWithDebInfo/seal-4.1.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1/SEALTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1/SEALTargets.cmake"
         "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/CMakeFiles/Export/38c274a671ce422ecc82c7dc6964cfb5/SEALTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1/SEALTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1/SEALTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1" TYPE FILE FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/CMakeFiles/Export/38c274a671ce422ecc82c7dc6964cfb5/SEALTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1" TYPE FILE FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/CMakeFiles/Export/38c274a671ce422ecc82c7dc6964cfb5/SEALTargets-debug.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1" TYPE FILE FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/CMakeFiles/Export/38c274a671ce422ecc82c7dc6964cfb5/SEALTargets-minsizerel.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1" TYPE FILE FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/CMakeFiles/Export/38c274a671ce422ecc82c7dc6964cfb5/SEALTargets-relwithdebinfo.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1" TYPE FILE FILES "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/CMakeFiles/Export/38c274a671ce422ecc82c7dc6964cfb5/SEALTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SEAL-4.1" TYPE FILE FILES
    "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/cmake/SEALConfig.cmake"
    "C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/cmake/SEALConfigVersion.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/willi/Documents/ISEN/M2/Projet start up/BlindGEN/seal_cpp_backend/build/native/seal/native/src/seal/cmake_install.cmake")

endif()

