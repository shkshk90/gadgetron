if (WIN32)
  add_definitions(-D__BUILD_GADGETRON_GADGET_GPUPMRI__)
  # The two flags below is to fix Windows problems in relation to multiple defined constructors in our headers
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE") 
  set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE") 
endif ()

add_library(gadgetron_gpuparallelmri SHARED
    gadgetron_gpupmri_export.h
    GenericReconJob.h
    gpuCgKtSenseGadget.h
    gpuCgSenseGadget.h
    gpuCgSpiritGadget.h
    gpuGenericSensePrepGadget.h
    gpuSbSenseGadget.h
    gpuCgSenseGadget.cpp.sycl.cpp
    gpuCgKtSenseGadget.cpp.sycl.cpp
    gpuSbSenseGadget.cpp.sycl.cpp
    gpuGenericSensePrepGadget.cpp.sycl.cpp
    gpuCgSpiritGadget.cpp.sycl.cpp
    gpuSenseGadget.cpp.sycl.cpp
    gpuBufferSensePrepGadget.cpp
    gpuOsSenseGadget.cpp.sycl.cpp
    gpuNlcgSenseGadget.cpp.sycl.cpp
    gpuLALMSenseGadget.cpp.sycl.cpp
  )

set_target_properties(gadgetron_gpuparallelmri PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_compile_options(gadgetron_gpuparallelmri PRIVATE ${SYCL_COMPILE_FLAGS})
target_link_options(gadgetron_gpuparallelmri PRIVATE ${SYCL_LINK_FLAGS})

target_link_libraries(gadgetron_gpuparallelmri
  gadgetron_core
  gadgetron_mricore
  gadgetron_toolbox_log
  gadgetron_toolbox_cpucore
  gadgetron_toolbox_gpu)

target_link_libraries(gadgetron_gpuparallelmri gadgetron_toolbox_gpu)

target_include_directories(gadgetron_gpuparallelmri
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})



install(TARGETS gadgetron_gpuparallelmri
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)

install (FILES  gadgetron_gpupmri_export.h
                GenericReconJob.h
                gpuCgKtSenseGadget.h
                gpuCgSenseGadget.h
                gpuCgSpiritGadget.h
                gpuGenericSensePrepGadget.h
                gpuSbSenseGadget.h
                gpuSenseGadget.h
                gpuBufferSensePrepGadget.h
                gpuOsSenseGadget.h
                gpuLALMSenseGadget.h
                gpuNlcgSenseGadget.h
                DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

add_subdirectory(config)

set(GADGETRON_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${GADGETRON_BUILD_RPATH}" PARENT_SCOPE)