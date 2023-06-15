add_library(${PROJECT_NAME}.scan_context
        scan_context.cc
        )

target_link_libraries(${PROJECT_NAME}.scan_context
        glog gflags ${PROJECT_NAME}.common
        )

add_executable(run_scan_context run_scan_context.cc)
target_link_libraries(run_scan_context
        ${PROJECT_NAME}.scan_context
        ${PCL_LIBRARIES}
        ${GLOG_LIBRARIES}
        gflags
        )
        