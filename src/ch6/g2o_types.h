//
// Created by xiang on 2022/3/22.
//

#ifndef SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>

#include <glog/logging.h>
#include <opencv2/core.hpp>

#include "common/eigen_types.h"
#include "common/math_utils.h"

namespace sad {

class VertexSE2 : public g2o::BaseVertex<3, SE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setToOriginImpl() override { _estimate = SE2(); }
    void oplusImpl(const double* update) override {
        _estimate.translation()[0] += update[0];
        _estimate.translation()[1] += update[1];
        _estimate.so2() = _estimate.so2() * SO2::exp(update[2]);
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }
};

/**Edge for Point to Point residual */
class EdgeICP_P2P : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeICP_P2P() {}
    EdgeICP_P2P(double rho_, double r_, Vec2d q_i_): rho(rho_), r(r_), q_i(q_i_){}

    void computeError() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double x_ = current_pose->estimate().translation()[0];
        double y_ = current_pose->estimate().translation()[1];
        double theta_ = current_pose->estimate().so2().log();
        Vec2d p_i = {x_ + r * std::cos(rho+theta_), y_ + r * std::sin(rho+theta_)};
        _error = p_i - q_i;
    }

    // TODO jacobian
    void linearizeOplus() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double theta_ = current_pose->estimate().so2().log();
        _jacobianOplusXi << 1, 0, 0, 1, -r * std::sin(rho + theta_), r * std::cos(rho + theta_);
    }
    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
        double rho;
        double r;
        double theta_, x_, y_;
        Vec2d q_i;
};

class EdgeICP_P2L : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeICP_P2L() {}
    EdgeICP_P2L(double rho_, double r_, Vec3d abc_): rho(rho_), r(r_), abc(abc_){}

    void computeError() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double x_ = current_pose->estimate().translation()[0];
        double y_ = current_pose->estimate().translation()[1];
        double theta_ = current_pose->estimate().so2().log();
        Vec2d p_i = {x_ + r * std::cos(rho+theta_), y_ + r * std::sin(rho+theta_)};
        _error[0] = p_i[0] * abc[0] + p_i[1] * abc[1] + abc[2];
    }

    // TODO jacobian
    void linearizeOplus() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double theta_ = current_pose->estimate().so2().log();
        _jacobianOplusXi << abc[0], abc[1], -abc[0]*r*std::sin(theta_+rho) + abc[1]*r*std::cos(theta_+rho);
    }
    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
        double rho;
        double r;
        Vec3d abc;
};

class EdgeSE2LikelihoodFiled : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2LikelihoodFiled(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
        : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {}

    /// 判定此条边是否在field image外面
    bool IsOutSide() {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2i pf = (pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2)).cast<int>();  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            return false;
        } else {
            return true;
        }
    }

    void computeError() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            _error[0] = math::GetPixelValue<float>(field_image_, pf[0], pf[1]);
        } else {
            _error[0] = 0;
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        float theta = pose.so2().log();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            // 图像梯度
            float dx = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0] + 1, pf[1]) -
                              math::GetPixelValue<float>(field_image_, pf[0] - 1, pf[1]));
            float dy = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0], pf[1] + 1) -
                              math::GetPixelValue<float>(field_image_, pf[0], pf[1] - 1));

            _jacobianOplusXi << resolution_ * dx, resolution_ * dy,
                -resolution_ * dx * range_ * std::sin(angle_ + theta) +
                    resolution_ * dy * range_ * std::cos(angle_ + theta);
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
    const cv::Mat& field_image_;
    double range_ = 0;
    double angle_ = 0;
    float resolution_ = 10.0;
    inline static const int image_boarder_ = 10;
};

/**
 * SE2 pose graph使用
 * error = v1.inv * v2 * meas.inv
 */
class EdgeSE2 : public g2o::BaseBinaryEdge<3, SE2, VertexSE2, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2() {}

    void computeError() override {
        VertexSE2* v1 = (VertexSE2*)_vertices[0];
        VertexSE2* v2 = (VertexSE2*)_vertices[1];
        _error = (v1->estimate().inverse() * v2->estimate() * measurement().inverse()).log();
    }

    // TODO jacobian

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
};


}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
