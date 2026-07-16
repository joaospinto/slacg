#include "tests/kkt_test_1/kkt_codegen.hpp"

#include <array>
#include <gtest/gtest.h>
#include <limits>

namespace slacg::test {

static_assert(expected_positive_inertia == x_dim);
static_assert(expected_negative_inertia == y_dim + z_dim);
static_assert(expected_zero_inertia == 0);

struct FactorWorkspace {
  std::array<double, L_nnz> L;
  std::array<double, core_dim> D_inv;
  std::array<double, border_solution_size> border_solution;
  std::array<double, border_factor_size> border_factor;
};

struct FactorResult {
  FactorStatus status;
  bool success;
};

FactorResult factor_with_border_diagonal(const double border_diagonal) {
  auto H_data = std::array<double, x_dim>{};
  auto C_data = std::array<double, x_dim * y_dim>{};
  auto G_data = std::array<double, x_dim * z_dim>{};
  auto s = std::array<double, z_dim>{};
  auto r1 = std::array<double, x_dim>{};
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  H_data.fill(1.0);
  H_data.back() = border_diagonal;
  s.fill(1.0);
  r1.fill(0.0);
  r2.fill(1e-3);
  r3.fill(1e-3);
  FactorWorkspace factor;

  const FactorStatus status = ldlt_factor_with_status(
      H_data.data(), C_data.data(), G_data.data(), s.data(), r1.data(),
      r2.data(), r3.data(), factor.L.data(), factor.D_inv.data(),
      factor.border_solution.data(), factor.border_factor.data());
  const bool success = ldlt_factor(
      H_data.data(), C_data.data(), G_data.data(), s.data(), r1.data(),
      r2.data(), r3.data(), factor.L.data(), factor.D_inv.data(),
      factor.border_solution.data(), factor.border_factor.data());
  return {status, success};
}

TEST(SLACG, Test) {
  auto H_data = std::array<double, x_dim>{};
  auto C_data = std::array<double, x_dim * y_dim>{};
  auto G_data = std::array<double, x_dim * z_dim>{};
  auto s = std::array<double, z_dim>{};
  H_data.fill(1.0);
  C_data.fill(1.0);
  G_data.fill(1.0);
  s.fill(1.0);
  auto r1 = std::array<double, x_dim>{};
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  for (int i = 0; i < x_dim; ++i) {
    r1[i] = 1e-3 + 0.01 * static_cast<double>(i + 1);
  }
  r2.fill(1e-3);
  r3.fill(1e-3);

  FactorWorkspace factor;

  auto b = std::array<double, dim>{};
  b.fill(1.0);

  std::array<double, dim> x;

  EXPECT_TRUE(ldlt_factor(H_data.data(), C_data.data(), G_data.data(), s.data(),
                          r1.data(), r2.data(), r3.data(), factor.L.data(),
                          factor.D_inv.data(), factor.border_solution.data(),
                          factor.border_factor.data()));
  EXPECT_EQ(ldlt_factor_with_status(H_data.data(), C_data.data(), G_data.data(),
                                    s.data(), r1.data(), r2.data(), r3.data(),
                                    factor.L.data(), factor.D_inv.data(),
                                    factor.border_solution.data(),
                                    factor.border_factor.data()),
            FactorStatus::kSuccess);

  ldlt_solve(factor.L.data(), factor.D_inv.data(),
             factor.border_solution.data(), factor.border_factor.data(),
             b.data(), x.data());

  std::array<double, dim> y;

  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = -b[i];
  }

  double *x_x = &x[0];
  double *x_y = &x[x_dim];
  double *x_z = &x[x_dim + y_dim];
  double *y_x = &y[0];
  double *y_y = &y[x_dim];
  double *y_z = &y[x_dim + y_dim];

  slacg::test::add_Kx_to_y(H_data.data(), C_data.data(), G_data.data(),
                           s.data(), r1.data(), r2.data(), r3.data(), x_x, x_y,
                           x_z, y_x, y_y, y_z);

  for (std::size_t i = 0; i < y.size(); ++i) {
    EXPECT_NEAR(y[i], 0.0, 1e-11);
  }
}

TEST(SLACG, DetectsWrongInertia) {
  auto H_data = std::array<double, x_dim>{};
  auto C_data = std::array<double, x_dim * y_dim>{};
  auto G_data = std::array<double, x_dim * z_dim>{};
  auto s = std::array<double, z_dim>{};
  H_data.fill(-100.0);
  C_data.fill(1.0);
  G_data.fill(1.0);
  s.fill(1.0);
  auto r1 = std::array<double, x_dim>{};
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  r2.fill(1e-3);
  r3.fill(1e-3);
  r1.fill(1e-3);

  FactorWorkspace factor;

  EXPECT_FALSE(ldlt_factor(
      H_data.data(), C_data.data(), G_data.data(), s.data(), r1.data(),
      r2.data(), r3.data(), factor.L.data(), factor.D_inv.data(),
      factor.border_solution.data(), factor.border_factor.data()));
  EXPECT_EQ(ldlt_factor_with_status(H_data.data(), C_data.data(), G_data.data(),
                                    s.data(), r1.data(), r2.data(), r3.data(),
                                    factor.L.data(), factor.D_inv.data(),
                                    factor.border_solution.data(),
                                    factor.border_factor.data()),
            FactorStatus::kWrongInertia);
}

TEST(SLACG, DetectsNonFinitePivot) {
  auto H_data = std::array<double, x_dim>{};
  auto C_data = std::array<double, x_dim * y_dim>{};
  auto G_data = std::array<double, x_dim * z_dim>{};
  auto s = std::array<double, z_dim>{};
  H_data.fill(std::numeric_limits<double>::infinity());
  C_data.fill(1.0);
  G_data.fill(1.0);
  s.fill(1.0);
  auto r1 = std::array<double, x_dim>{};
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  r2.fill(1e-3);
  r3.fill(1e-3);
  r1.fill(1e-3);

  FactorWorkspace factor;

  EXPECT_FALSE(ldlt_factor(
      H_data.data(), C_data.data(), G_data.data(), s.data(), r1.data(),
      r2.data(), r3.data(), factor.L.data(), factor.D_inv.data(),
      factor.border_solution.data(), factor.border_factor.data()));
  EXPECT_EQ(ldlt_factor_with_status(H_data.data(), C_data.data(), G_data.data(),
                                    s.data(), r1.data(), r2.data(), r3.data(),
                                    factor.L.data(), factor.D_inv.data(),
                                    factor.border_solution.data(),
                                    factor.border_factor.data()),
            FactorStatus::kNonFinitePivot);
}

TEST(SLACG, DetectsZeroPivot) {
  auto H_data = std::array<double, x_dim>{};
  auto C_data = std::array<double, x_dim * y_dim>{};
  auto G_data = std::array<double, x_dim * z_dim>{};
  auto s = std::array<double, z_dim>{};
  H_data.fill(1.0);
  C_data.fill(0.0);
  G_data.fill(0.0);
  s.fill(0.0);
  auto r1 = std::array<double, x_dim>{};
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  r2.fill(0.0);
  r3.fill(0.0);
  r1.fill(0.0);

  FactorWorkspace factor;

  EXPECT_FALSE(ldlt_factor(
      H_data.data(), C_data.data(), G_data.data(), s.data(), r1.data(),
      r2.data(), r3.data(), factor.L.data(), factor.D_inv.data(),
      factor.border_solution.data(), factor.border_factor.data()));
  EXPECT_EQ(ldlt_factor_with_status(H_data.data(), C_data.data(), G_data.data(),
                                    s.data(), r1.data(), r2.data(), r3.data(),
                                    factor.L.data(), factor.D_inv.data(),
                                    factor.border_solution.data(),
                                    factor.border_factor.data()),
            FactorStatus::kZeroPivot);
}

TEST(SLACG, DetectsWrongBorderInertia) {
  const FactorResult result = factor_with_border_diagonal(-1.0);
  EXPECT_EQ(result.status, FactorStatus::kWrongInertia);
  EXPECT_FALSE(result.success);
}

TEST(SLACG, DetectsNonFiniteBorderPivot) {
  const FactorResult result =
      factor_with_border_diagonal(std::numeric_limits<double>::infinity());
  EXPECT_EQ(result.status, FactorStatus::kNonFinitePivot);
  EXPECT_FALSE(result.success);
}

TEST(SLACG, DetectsZeroBorderPivot) {
  const FactorResult result = factor_with_border_diagonal(0.0);
  EXPECT_EQ(result.status, FactorStatus::kZeroPivot);
  EXPECT_FALSE(result.success);
}

} // namespace slacg::test
