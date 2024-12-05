#include "ldlt_codegen_for_example.hpp"

#include <array>
#include <gtest/gtest.h>

namespace slacg::example {

TEST(SLACG, Example) {
  constexpr auto Ax = std::array{46., 54., 67., 63., 78., 94.};
  constexpr auto b = std::array{180., 223., 266.};

  std::array<double, 3> x;

  slacg::example::ldlt_solve(Ax.data(), b.data(), x.data());

  EXPECT_NEAR(x[0], 0.0, 1e-12);
  EXPECT_NEAR(x[1], 1.0, 1e-12);
  EXPECT_NEAR(x[2], 2.0, 1e-12);
}

}  // namespace slacg::example
