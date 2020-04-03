#include <vector>
#include <Eigen/Dense>

#include "geometry/curvilinear_coordinate_system.h"

using namespace geometry;


int main() {
    EigenPolyline reference_path(5);
    reference_path.push_back(Eigen::Vector2d(-1.2, 0.0));
    reference_path.push_back(Eigen::Vector2d(0.0, 0.0));
    reference_path.push_back(Eigen::Vector2d(3.0, 0.5));
    reference_path.push_back(Eigen::Vector2d(4.5, 0.0));
    reference_path.push_back(Eigen::Vector2d(6.5, 0.0));

    std::shared_ptr<CurvilinearCoordinateSystem> cosy = std::make_shared<CurvilinearCoordinateSystem>(reference_path);

}
