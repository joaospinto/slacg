#include "tests/kkt_test_no_inequalities/kkt_codegen.hpp"

#include <gtest/gtest.h>

namespace slacg::test {

TEST(KKTCodegen, SupportsNoInequalities) {
  static_assert(x_dim == 2);
  static_assert(y_dim == 1);
  static_assert(z_dim == 0);
  EXPECT_EQ(dim, 3);
}

} // namespace slacg::test
