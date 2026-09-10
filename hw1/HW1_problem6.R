set.seed(123)

Nmax <- 1e6

x <- runif(Nmax)
y <- runif(Nmax)
z <- runif(Nmax)

success <- (x^2 + y^2 < z) & (z^2 > x*y)

# Cumulative Monte Carlo estimate
cum_est <- cumsum(success) / seq_len(Nmax)

# Log-spaced sample sizes
Nvals <- unique(round(10^seq(1, 6, length.out = 200)))

analytic <- 23*pi/192

plot(
  Nvals,
  cum_est[Nvals],
  type = "l",
  log = "x",
  xlab = "Sample size N",
  ylab = "Monte Carlo estimate",
  main = "Monte Carlo Estimate of Probability"
)

abline(
  h = analytic,
  lty = 2,
  lwd = 2
)

legend(
  "topright",
  legend = c("Monte Carlo", "Analytic = 23pi/192"),
  lty = c(1, 2),
  lwd = c(1, 2)
)

analytic
mean(success)
