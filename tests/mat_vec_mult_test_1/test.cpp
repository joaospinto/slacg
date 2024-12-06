#include "tests/mat_vec_mult_test_1/mat_vec_mult_codegen.hpp"

#include <array>
#include <gtest/gtest.h>

namespace slacg::test {

TEST(SLACG, Test) {
  constexpr auto A_data = std::array{
    10., 20., 30., 40.,  1., 11., 21., 31., 41.,  2., 12., 22., 32.,
    42.,  3., 13., 23., 33., 43.,  4., 14., 24., 34., 44.,  5., 15.,
    25., 35., 45.,  6., 16., 26., 36., 46.,  7., 17., 27., 37., 47.,
    8., 18., 28., 38., 48.,  9., 19., 29., 39., 49.};

  {
    std::array<double, 10> x{0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::array<double, 5> y{-285., -735., -1185., -1635., -2085.};
    add_Ax_to_y(A_data.data(), x.data(), y.data());
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_NEAR(y[0], 0.0, 1e-12);
    }
  }

  {
    std::array<double, 5> x{0., 1., 2., 3., 4.};
    std::array<double, 10> y{-300., -310., -320., -330., -340., -350., -360., -370., -380., -390.};
    add_ATx_to_y(A_data.data(), x.data(), y.data());
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_NEAR(y[0], 0.0, 1e-12);
    }
  }
}

}  // namespace slacg::test
