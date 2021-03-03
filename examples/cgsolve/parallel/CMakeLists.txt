add_executable(cgsolve cgsolve.cpp)
target_link_libraries(cgsolve PRIVATE Kokkos::kokkosremote)
target_include_directories(cgsolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pipeline_cgsolve pipeline_cgsolve.cpp)
target_link_libraries(pipeline_cgsolve PRIVATE Kokkos::kokkosremote)
target_include_directories(pipeline_cgsolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pipeline_cgsolve_mpi pipeline_cgsolve_mpi.cpp)
target_link_libraries(pipeline_cgsolve_mpi PRIVATE Kokkos::kokkosremote)
target_include_directories(pipeline_cgsolve_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cgsolve_mpi cgsolve_mpi.cpp)
target_link_libraries(cgsolve_mpi PRIVATE Kokkos::kokkosremote)
target_include_directories(cgsolve_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(sstep_cgsolve_mpi sstep_cgsolve_mpi.cpp)
target_link_libraries(sstep_cgsolve_mpi PRIVATE Kokkos::kokkosremote)
target_include_directories(sstep_cgsolve_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
