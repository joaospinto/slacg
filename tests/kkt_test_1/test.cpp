#include "tests/kkt_test_1/kkt_codegen.hpp"

#include <array>
#include <gtest/gtest.h>

namespace slacg::test {

TEST(SLACG, Test) {
  auto H_data = std::array<double, x_dim>{};
  auto C_data = std::array<double, x_dim * y_dim>{};
  auto G_data = std::array<double, x_dim * z_dim>{};
  auto s = std::array<double, z_dim>{};
  H_data.fill(1.0);
  C_data.fill(1.0);
  G_data.fill(1.0);
  s.fill(1.0);
  constexpr double r1 = 1e-3;
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  r2.fill(1e-3);
  r3.fill(1e-3);

  std::array<double, L_nnz> L_data;
  std::array<double, dim> D_inv;

  auto b = std::array<double, dim>{};
  b.fill(1.0);

  std::array<double, dim> x;

  EXPECT_TRUE(ldlt_factor(H_data.data(), C_data.data(), G_data.data(), s.data(),
                          r1, r2.data(), r3.data(), L_data.data(),
                          D_inv.data()));

  ldlt_solve(L_data.data(), D_inv.data(), b.data(), x.data());

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
                           s.data(), r1, r2.data(), r3.data(), x_x, x_y, x_z,
                           y_x, y_y, y_z);

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
  constexpr double r1 = 1e-3;
  auto r2 = std::array<double, y_dim>{};
  auto r3 = std::array<double, z_dim>{};
  r2.fill(1e-3);
  r3.fill(1e-3);

  std::array<double, L_nnz> L_data;
  std::array<double, dim> D_inv;

  EXPECT_FALSE(ldlt_factor(H_data.data(), C_data.data(), G_data.data(),
                           s.data(), r1, r2.data(), r3.data(), L_data.data(),
                           D_inv.data()));
}

} // namespace slacg::test
