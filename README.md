# analyzing-various-case-scenarios

Overview

Applying what I've learned to analyze case scenarios from political campaigning, the stock market and the entertainment industry.

Following the 3 questions provided in the assignment. I will explain code output, observed data, and explain various plots.

Question 1: Bayesian Decision Making in Election Strategy

https://drive.google.com/file/d/16G8xzJB6OWvvMinCom_DneKBPLrt36VG/view?usp=sharing

1. Plot the prior distribution of P(vote for candidate). How does the prior reflect the strategist's belief about the likelihood of winning the state?

The prior reflects a moderately optimistic belief about the strategist’s chances of winning the state. The prior mean of 0.60 indicates that, before reviewing any new polling data, the strategist believes there is roughly a 60% chance the state will vote for their candidate. The mode, approximately 0.67, shows that the most plausible value for the win probability is around two-thirds, further reinforcing this optimistic outlook. However, the wide 95% credible interval reveals substantial uncertainty, indicating that while the strategist expects to win the state, they acknowledge a meaningful possibility of both significantly lower or substantially higher levels of support. Overall, the prior expresses confidence leaning toward a favorable outcome, but with enough uncertainty to allow new data to meaningfully update the belief.


2. After receiving new polling data that suggests a 60% probability of the state voting for the candidate, update the prior using Bayes' Theorem to obtain the posterior distribution of P(vote for candidate)  

https://drive.google.com/file/d/18RR2rPwRzm5X_YFS-YeZOUQD6xA2mF_g/view?usp=sharing

After incorporating the polling data indicating 60% support from a sample of 50 respondents, the prior is updated through Bayes’ Theorem to produce a new posterior distribution for P(vote for candidate). Because the likelihood corresponds to observing k=30 “successes” out of 50, the posterior becomes strongly influenced by the poll. The resulting distribution is substantially more concentrated around values near 0.60, reflecting the weight of the new evidence. Compared to the prior, which was moderately optimistic but wide, the posterior is sharper and more confident, showing that the strategist now believes the true probability of winning the state is very likely to be close to the polling estimate, with far less uncertainty.

 3. Create a function expected_loss(loss_matrix, posterior) that takes as input the posterior distribution and a 2x2 cost matrix, and returns the expected loss for each of the 2 possible actions. Which action minimizes the expected loss?

https://drive.google.com/file/d/1ERGR-4kseHJoWkUdHXgrdxZ5GSuoxJ53/view?usp=sharing

Using the posterior distribution updated from the polling data and applying the expected_loss function with the specified 2×2 loss matrix, we obtain the expected loss for each possible action. Action A (allocating campaign resources) has an expected loss of 23 units, reflecting the combined risk of spending resources unnecessarily or benefiting only slightly when the state already supports the candidate. In contrast, Action B (not allocating resources) has a lower expected loss of 12 units, since the only penalty comes from missing an opportunity if the state votes for the candidate. The expected loss for Action B is considerably smaller, so the strategist should choose Action B because it is optimal not to allocate additional campaign resources based on the current posterior belief about winning the state.

Question 2: Sequential Stock Price Prediction

1:  Run SIS without resampling and plot at the same graph the observed prices y(t) and the estimated S(t), i.e., the particle mean.

https://drive.google.com/file/d/1BxFMHGQ9lMOH1YojYSga1wwNTUuYEFqO/view?usp=sharing

Observing the graph, the SIS estimate tracks the observed prices reasonably well in the early part of the series. During the first 15–20 days, the orange line is very close to the blue line, capturing both the overall level and the short-term fluctuations in the stock price. This indicates that initially, the particle filter is effectively estimating the true underlying stock price based on all the observations up to that point, with the posterior mean aligning with the data. However, as time goes on, especially after about day 25–30, the differences become more noticeable. The estimated S(t) becomes visibly smoother and lags behind the sharp downward movements in the observed prices. We can see in the later days, the orange line tends to stay above the blue line, so the filter underestimates how far and how quickly the actual price drops. This pattern aligns with weight degeneracy in SIS without resampling. Over many steps, a small number of particles carry almost all the weight, causing the effective sample size to decrease and the filter to respond less to new observations. Consequently, the posterior approximation becomes biased and overly smooth, resulting in an estimated price path that doesn't fully follow the significant downward moves seen in the actual TSLA closing prices. 

2: Run SIS again with the systematic_resampling(particles, weights, N) and plot again y(t) and S(t) on the same graph. What do you observe? 

https://drive.google.com/file/d/1Cc9olDdFribgpAFEQuB1gsMbU3gRGEmi/view?usp=sharing


With resampling turned on, the particle filter now tracks the Tesla stock price much more closely throughout the entire time period, not just at the beginning. Observing the plot, the orange line (the particle-mean estimate of the stock price) almost lies on top of the blue line (the observed closing prices) over the whole series. In the early days the behavior is similar to the no-resampling case and the estimate follows the observed prices very well. However, the key difference appears in the later days as the estimate now continues to move up and down with the observations, capturing the sharp downward trend and the small rebounds near the end, much more accurately compared to the plot with the resampling, where the estimate tended to stay above the true price when the stock dropped. This indicates that resampling is preventing weight degeneracy. Instead of a few particles dominating and making the filter insensitive to new information, the resampling step regularly refreshes the particles in regions supported by the data. As a result, the effective number of particles remains high, and the filter keeps providing a responsive, well-calibrated estimate of the underlying stock price. Overall, we can see that when resampling is used, the sequential importance sampling algorithm gives a much better approximation to the underlying stock price process over time, especially in the later part of the series, where the version without resampling started to drift and lag behind the actual prices.

Question 3: Comparing Variational Inference and MCMC

1: Implement vi_model(iterations = 50000, n_samples = 2000),a function that will learn the same model but using variational inference, with parameters provided as arguments. The function will return the run time, as well as the mean of the posteriors for the 2 coefficients (beta1 and beta2) of the model. You should compute the run time for 5 different numbers of iterations (5000, 10000, 25000, 50000, 100000) using fullrank_advi (method="fullrank_advi"), and plot the results together with the run time for the MCMC (the latter can be a horizontal line on the graph).

https://drive.google.com/file/d/1jpxzISNnbR3bde_HITnKnAqYEg3RpbRA/view?usp=sharing

To compare Variational Inference (VI) with Markov Chain Monte Carlo (MCMC), I implemented the same Bayesian linear regression model with both methods using IMDB ratings as the outcome and a movie’s runtime and revenue-to-budget ratio as predictors. Running the MCMC sampler produced draws from the full posterior but required about 63 seconds, reflecting the computational expense associated with generating thousands of posterior samples. I then trained the same model using full-rank ADVI and evaluated VI at five different optimization budgets (5000, 10000, 25000, 50000, and 100000 iterations). The resulting VI runtimes ranged from about 3 seconds at 5,000 iterations to 29 seconds at 100,000 iterations, demonstrating that VI is consistently much faster than MCMC across all iteration counts.

The runtime plot illustrates this contrast where the VI curve increases gradually and almost linearly with the number of iterations, while the MCMC runtime appears as a horizontal dashed line far above the entire VI curve. This visually emphasizes that even the slowest VI setting remained less than half the time of MCMC, and in most cases VI ran more than 10x faster.

In addition to runtime, I compared posterior mean estimates for the regression coefficients across VI configurations. At low iteration counts (5000 and 10000), the VI posterior means varied substantially from the MCMC estimates, reflecting a less accurate approximation. However, as the number of iterations increased, especially at 25,000 iterations and above, the VI posterior means stabilized and converged toward the MCMC values. This pattern highlights the trade-off between the two methods, in which the MCMC delivers the most accurate and reliable posterior estimates but requires more computation, while VI provides faster results but only approaches MCMC accuracy when given a large number of optimization steps.

question 2:  Using the same function, I will plot the difference between the posterior mean for beta2 using the MCMC (full posterior) and its approximation for different numbers of iterations and explain what I observe.

https://drive.google.com/file/d/1OGzxBbAaVVdPl24QsK1MgtcEFrMilJsX/view?usp=sharing

https://drive.google.com/file/d/1H3k9er5OEMjz9Bl_Tc-qR0vttB4Q-gv7/view?usp=sharing

https://drive.google.com/file/d/18NBf_4gzO40fuP0l7jwyuGMPqa6hxicR/view?usp=sharing

When comparing the posterior mean of Beta2 estimated by Variational Inference (VI) to the corresponding MCMC posterior mean, the plots show a clear pattern in how VI’s approximation improves as the number of iterations increases. At low iteration counts, such as 5,000 and 10,000 iterations, the VI estimates for Beta2 differ from the MCMC benchmark. This is reflected in the difference plot, where the absolute error is largest for these two settings, reaching values above 0.12. As the number of iterations increases to 25,000, the VI estimate moves much closer to the MCMC value, reducing the difference dramatically. Once the number of iterations reaches 50,000 and 100,000, the VI estimate converges to the MCMC mean, and the absolute difference becomes nearly zero. These results illustrate the expected behavior of VI. While VI runs much faster than MCMC, its accuracy depends on the number of optimization steps. With too few iterations, VI struggles to match the full posterior produced by MCMC, but with sufficient iterations, it can approximate the MCMC estimate extremely closely. In this case, VI begins producing reliable estimates of Beta2 only after around 25,000 iterations, and by 50,000 iterations it basically matches the MCMC estimate. Observing the plots, we can confirm that VI converges toward the full posterior solution as the optimization budget increases, eventually becoming nearly identical to the MCMC result for this coefficient.
