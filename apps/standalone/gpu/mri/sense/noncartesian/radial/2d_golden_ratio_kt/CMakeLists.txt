add_executable(ktsense_radial_2d  main.cpp.sycl.cpp)

target_link_libraries(ktsense_radial_2d gadgetron_toolbox_gpu gadgetron_toolbox_hostutils)

install(TARGETS ktsense_radial_2d DESTINATION bin COMPONENT main)
