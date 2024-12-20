#include "tests/ldlt_test_1/ldlt_codegen.hpp"
#include "tests/ldlt_test_1/mat_vec_mult_codegen.hpp"

#include <array>
#include <gtest/gtest.h>

namespace slacg::test {

TEST(SLACG, Test) {
  constexpr auto A_data = std::array{46., 54., 67., 63., 78., 94.};
  constexpr auto b = std::array{180., 223., 266.};

  std::array<double, 3> LT_data;
  std::array<double, 3> D_diag;
  std::array<double, 3> x;

  ldlt_factor(A_data.data(), LT_data.data(), D_diag.data());
  ldlt_solve(LT_data.data(), D_diag.data(), b.data(), x.data());

  EXPECT_NEAR(x[0], 0.0, 1e-12);
  EXPECT_NEAR(x[1], 1.0, 1e-12);
  EXPECT_NEAR(x[2], 2.0, 1e-12);

  std::array<double, 3> y{0.0, 0.0, 0.0};

  add_upper_symmetric_Ax_to_y(A_data.data(), x.data(), y.data());

  EXPECT_NEAR(y[0] - b[0], 0.0, 1e-12);
  EXPECT_NEAR(y[1] - b[1], 0.0, 1e-12);
  EXPECT_NEAR(y[2] - b[2], 0.0, 1e-12);
}

}  // namespace slacg::test
