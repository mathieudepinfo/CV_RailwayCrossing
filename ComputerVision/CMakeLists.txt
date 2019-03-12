# CMakeList.txt : projet CMake pour ComputerVision, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)


include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${LIBD} ${LIB})

add_executable (ComputerVision "Histogram.cpp" "Histogram.h" "ComputerVision.cpp" "ComputerVision.h" "CCTVsystem.h" "CCTVsystem.cpp")

target_link_libraries( ComputerVision debug ${OpenCV_LIBD} optimized ${OpenCV_LIB} )
# Ajoutez une source à l'exécutable de ce projet.


# TODO: Ajoutez des tests et installez des cibles si nécessaire.
