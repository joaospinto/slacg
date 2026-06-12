#include "benchmarks/kernels/gtwg_codegen.hpp"
#include "benchmarks/kernels/kkt_codegen.hpp"
#include "benchmarks/kernels/ldlt_codegen.hpp"
#include "benchmarks/kernels/mat_vec_mult_codegen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

template <typename Fn>
double benchmark_ns_per_call(const int iterations, Fn fn) {
  fn();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  const auto end = std::chrono::steady_clock::now();
  const auto total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(total_ns) / static_cast<double>(iterations);
}

int iterations_from_args(const int argc, char **argv) {
  if (argc < 2) {
    return 10000;
  }
  return std::max(1, std::atoi(argv[1]));
}

template <std::size_t N>
void fill_data(std::array<double, N> &data) {
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = 1.0 + static_cast<double>(i % 7) * 0.01;
  }
}

void print_result(const std::string &name, const int iterations,
                  const double ns_per_call) {
  std::cout << name << ": " << ns_per_call << " ns/call"
            << " (" << iterations << " iterations)\n";
}

void run_ldlt_factor(const int iterations) {
  std::array<double, 170> A_data{};
  std::array<double, slacg::bench::ldlt::L_nnz> LT_data{};
  std::array<double, slacg::bench::ldlt::dim> D_inv{};
  fill_data(A_data);

  print_result("ldlt_factor", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 slacg::bench::ldlt::ldlt_factor(A_data.data(), LT_data.data(),
                                                  D_inv.data());
               }));
}

void run_ldlt_solve(const int iterations) {
  std::array<double, 170> A_data{};
  std::array<double, slacg::bench::ldlt::L_nnz> LT_data{};
  std::array<double, slacg::bench::ldlt::dim> D_inv{};
  std::array<double, slacg::bench::ldlt::dim> b{};
  std::array<double, slacg::bench::ldlt::dim> x{};
  fill_data(A_data);
  fill_data(b);
  slacg::bench::ldlt::ldlt_factor(A_data.data(), LT_data.data(),
                                  D_inv.data());

  print_result("ldlt_solve", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 slacg::bench::ldlt::ldlt_solve(LT_data.data(), D_inv.data(),
                                                b.data(), x.data());
               }));
}

void run_matvec(const int iterations) {
  std::array<double, 240> A_data{};
  std::array<double, 48> x{};
  std::array<double, 72> y{};
  fill_data(A_data);
  fill_data(x);

  print_result("matvec_add_Ax", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 y.fill(0.0);
                 slacg::bench::matvec::add_Ax_to_y(A_data.data(), x.data(),
                                                   y.data());
               }));
}

void run_gtwg(const int iterations) {
  std::array<double, 192> G_data{};
  std::array<double, 64> w{};
  std::array<double, 105> gt_w_g{};
  fill_data(G_data);
  fill_data(w);

  print_result("gt_w_g", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 slacg::bench::gtwg::gt_w_g(G_data.data(), w.data(), 1e-3,
                                            gt_w_g.data());
               }));
}

void run_kkt_factor(const int iterations) {
  std::array<double, 90> H_data{};
  std::array<double, 96> C_data{};
  std::array<double, 120> G_data{};
  std::array<double, slacg::bench::kkt::z_dim> w{};
  std::array<double, slacg::bench::kkt::y_dim> r2{};
  std::array<double, slacg::bench::kkt::z_dim> r3{};
  std::array<double, slacg::bench::kkt::L_nnz> LT_data{};
  std::array<double, slacg::bench::kkt::dim> D_inv{};
  fill_data(H_data);
  fill_data(C_data);
  fill_data(G_data);
  fill_data(w);
  r2.fill(1e-3);
  r3.fill(1e-3);

  print_result("kkt_ldlt_factor", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 slacg::bench::kkt::ldlt_factor(
                     H_data.data(), C_data.data(), G_data.data(), w.data(),
                     1e-3, r2.data(), r3.data(), LT_data.data(),
                     D_inv.data());
               }));
}

void run_kkt_solve(const int iterations) {
  std::array<double, 90> H_data{};
  std::array<double, 96> C_data{};
  std::array<double, 120> G_data{};
  std::array<double, slacg::bench::kkt::z_dim> w{};
  std::array<double, slacg::bench::kkt::y_dim> r2{};
  std::array<double, slacg::bench::kkt::z_dim> r3{};
  std::array<double, slacg::bench::kkt::L_nnz> LT_data{};
  std::array<double, slacg::bench::kkt::dim> D_inv{};
  std::array<double, slacg::bench::kkt::dim> b{};
  std::array<double, slacg::bench::kkt::dim> x{};
  fill_data(H_data);
  fill_data(C_data);
  fill_data(G_data);
  fill_data(w);
  fill_data(b);
  r2.fill(1e-3);
  r3.fill(1e-3);
  slacg::bench::kkt::ldlt_factor(H_data.data(), C_data.data(), G_data.data(),
                                 w.data(), 1e-3, r2.data(), r3.data(),
                                 LT_data.data(), D_inv.data());

  print_result("kkt_ldlt_solve", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 slacg::bench::kkt::ldlt_solve(LT_data.data(), D_inv.data(),
                                               b.data(), x.data());
               }));
}

void run_kkt_add_Kx(const int iterations) {
  std::array<double, 90> H_data{};
  std::array<double, 96> C_data{};
  std::array<double, 120> G_data{};
  std::array<double, slacg::bench::kkt::z_dim> w{};
  std::array<double, slacg::bench::kkt::y_dim> r2{};
  std::array<double, slacg::bench::kkt::z_dim> r3{};
  std::array<double, slacg::bench::kkt::x_dim> x_x{};
  std::array<double, slacg::bench::kkt::y_dim> x_y{};
  std::array<double, slacg::bench::kkt::z_dim> x_z{};
  std::array<double, slacg::bench::kkt::x_dim> y_x{};
  std::array<double, slacg::bench::kkt::y_dim> y_y{};
  std::array<double, slacg::bench::kkt::z_dim> y_z{};
  fill_data(H_data);
  fill_data(C_data);
  fill_data(G_data);
  fill_data(w);
  fill_data(x_x);
  fill_data(x_y);
  fill_data(x_z);
  r2.fill(1e-3);
  r3.fill(1e-3);

  print_result("kkt_add_Kx_to_y", iterations,
               benchmark_ns_per_call(iterations, [&] {
                 y_x.fill(0.0);
                 y_y.fill(0.0);
                 y_z.fill(0.0);
                 slacg::bench::kkt::add_Kx_to_y(
                     H_data.data(), C_data.data(), G_data.data(), w.data(),
                     1e-3, r2.data(), r3.data(), x_x.data(), x_y.data(),
                     x_z.data(), y_x.data(), y_y.data(), y_z.data());
               }));
}

}  // namespace

int main(int argc, char **argv) {
  const int iterations = iterations_from_args(argc, argv);
  run_ldlt_factor(iterations);
  run_ldlt_solve(iterations);
  run_matvec(iterations);
  run_gtwg(iterations);
  run_kkt_factor(iterations);
  run_kkt_solve(iterations);
  run_kkt_add_Kx(iterations);
  return 0;
}
