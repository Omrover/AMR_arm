cmake_minimum_required(VERSION 3.22)
project(amr_arm_moveit_config NONE)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_srdf test/test_srdf.py TIMEOUT 120)
endif()

ament_package()
