find_package(ZFP)


if (ZFP_FOUND)
   add_definitions(-DGADGETRON_COMPRESSION_ZFP)
endif()

if (ZFP_FOUND)
   include_directories(${ZFP_INCLUDE_DIR})
endif()


if (NOT USE_AMALGAMATED)
    add_executable(gadgetron_ismrmrd_client gadgetron_ismrmrd_client.cpp)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(gadgetron_ismrmrd_client ISMRMRD::ISMRMRD Boost::program_options gadgetron_mricore )

    if (ZFP_FOUND)
       target_link_libraries(gadgetron_ismrmrd_client ${ZFP_LIBRARIES})
    endif ()

    install(TARGETS gadgetron_ismrmrd_client DESTINATION bin COMPONENT main)
else()
    add_executable(gadgetron_ismrmrd_client gadgetron_ismrmrd_client_amalgamated.cpp)
    if (ZFP_FOUND)
       target_link_libraries(gadgetron_ismrmrd_client ${GADGETRON_AMALGAMATED_LIBS} ${ZFP_LIBRARIES})
    else ()
      target_link_libraries(gadgetron_ismrmrd_client ${GADGETRON_AMALGAMATED_LIBS})
   endif ()
   set_source_files_properties(gadgetron_ismrmrd_client_amalgamated.cpp PROPERTIES COMPILE_FLAGS "-msse4.1")
  set_source_files_properties(gadgetron_ismrmrd_client_amalgamated.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    install(TARGETS gadgetron_ismrmrd_client DESTINATION bin COMPONENT main)
endif()
