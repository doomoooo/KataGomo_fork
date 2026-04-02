#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace schedlab {
  namespace poisson_detail {
    constexpr double gamma_eps = 1e-8;
    constexpr double gamma_min = 1e-300;
    constexpr int gamma_max_iter = 256;

    inline auto regularized_gamma_p_series(double a, double x) noexcept -> double {
      double term = 1.0 / a;
      double sum = term;
      double ap = a;
      for(int i = 0; i < gamma_max_iter; ++i) {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if(std::abs(term) <= std::abs(sum) * gamma_eps) {
          break;
        }
      }
      const double scale = std::exp(a * std::log(x) - x - std::lgamma(a));
      return std::clamp(sum * scale, 0.0, 1.0);
    }

    inline auto regularized_gamma_q_contfrac(double a, double x) noexcept -> double {
      double b = x + 1.0 - a;
      double c = 1.0 / gamma_min;
      double d = 1.0 / (std::abs(b) < gamma_min ? gamma_min : b);
      double h = d;
      for(int i = 1; i <= gamma_max_iter; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if(std::abs(d) < gamma_min) {
          d = gamma_min;
        }
        c = b + an / c;
        if(std::abs(c) < gamma_min) {
          c = gamma_min;
        }
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if(std::abs(delta - 1.0) <= gamma_eps) {
          break;
        }
      }
      const double scale = std::exp(a * std::log(x) - x - std::lgamma(a));
      return std::clamp(scale * h, 0.0, 1.0);
    }

    inline auto regularized_gamma_q(double a, double x) noexcept -> double {
      if(x <= 0.0) {
        return 1.0;
      }
      if(x < a + 1.0) {
        return 1.0 - regularized_gamma_p_series(a, x);
      }
      return regularized_gamma_q_contfrac(a, x);
    }
  }  // namespace poisson_detail

  // P(N < needed_requests), N ~ Poisson(mean_requests)。
  inline auto poisson_starvation_probability(std::uint64_t needed_requests, double mean_requests) noexcept
    -> double {
    if(needed_requests == 0) {
      return 0.0;
    }
    return poisson_detail::regularized_gamma_q(static_cast<double>(needed_requests), mean_requests);
  }
}  // namespace schedlab
