# ------ background ------
# Purpose: Rerun the ridge regression and XGBoost regression analyses 
# for the article "Auditory scene analysis in music and speech: individual differences in listening abilities"
# Authors: 
# Robin Hake1, Daniel Müllensiefen2,3 & Kai Siedenburg1,4
# 1 Dept. of Medical Physics and Acoustics, University of Oldenburg, Germany 
# 2 Institute for Systematic Musicology, University of Hamburg, Germany
# 3 Department of Psychology, Goldsmiths, University of London, UK
# 4 Signal Processing and Speech Communication Laboratory Graz University of Technology, Austria

# Submitted to Scientific Reports on: 18.02.2025

# Abstract 
# Auditory scene analysis (ASA) is the ability to organize complex auditory mixtures into
# meaningful events and streams and is fundamental for auditory perception of both speech
# and music. Individual differences in ASA are recognized in the literature, yet the
# factors driving this variability remain poorly understood. This study employs a novel
# music-based ASA task, the Musical Scene Analysis (MSA) test, alongside a speech-in-noise
# test, to examine the influence of hearing loss, age, working memory capacity (WMC), and
# musical training. Ninety-two participants were categorised into four groups: 31 older
# normal-hearing, 34 older hearing-impaired, 26 younger normal-hearing, and one younger
# hearing-impaired individual. Results reveal a moderate correlation between ASA
# performance in speech and music (r = -.5), suggesting shared underlying perceptual
# processes, yet the factors influencing individual differences varied across domains. A
# dual modeling approach using ridge regression and gradient-boosted decision trees
# identified hearing loss as the strongest predictor of speech-based ASA, with a weaker
# effect of age, while musical training and WMC had no impact. In contrast, musical
# training showed a substantial effect on musical ASA, alongside moderate effects of
# hearing loss and age, while WMC exhibited only a marginal, non-robust effect. These
# findings highlight both shared and domain-specific factors influencing ASA abilities in
# speech and music.



# ------ libraries ------
library(tidyverse)
library(caret)
library(xgboost) # for the gradient boosted decicion trees
library(dplyr)
library(tidyr)
library(tibble)
library(Hmisc)        # for summary statistics
library(psych)
library(janitor)
library(SHAPforxgboost) # for SHAP calculations
library(kernelshap)   # for SHAP calculations
library(shapviz)      # for SHAP visualisations
library(glmnet) #for ridge regression;
library(boot) #for bootstrap resampling;
library(ggplot2)
library(ggridges)
library(reshape2)
library(rlang)    # for tidy evaluation
library(dplyr)
library(ggplot2)
library(magrittr)

# ------ preps ------
### Kill global environment (uncomment if needed)
# remove(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
wd <- print(getwd())

# ------ Data inclusion ------
# Embed the complete dataset as a multi-line string.
# Note: 'NA' denotes missing values.
data_text <- "
   p_id   gender age BDS_average GMS_training   PTA_full         MSA        SRT group
1     1 Weiblich  52  0.59126984     4.857143  6.4285714  0.12296764 -6.0414167   oNH
2     2 Weiblich  55  0.46269841     1.285714  5.0000000  0.45641002 -5.3659572   oNH
3     3 Männlich  66  0.60714286     3.142857 40.0000000 -0.04787417 -2.1474770   oHI
4     4 Männlich  70  0.45039683     3.857143 10.7142857  0.29873897 -5.2490600   oNH
5     5 Weiblich  26  0.69801587     1.285714 -6.4285714  0.28904750 -5.5870298   yNH
6     6 Weiblich  53  0.67698413     4.428571 12.8571429  0.22265280 -5.3353614   oNH
7     7 Weiblich  32  0.77619048     1.714286  4.2857143  0.39197479 -6.2179690   yNH
8     8 Weiblich  69  0.66547619     5.428571 18.1818182  0.44232283 -4.4892350   oNH
9     9 Weiblich  69  0.55753968     1.428571 42.1428571 -0.32392034 -2.7060888   oHI
10   10 Männlich  53  0.55396825     1.285714  7.8571429  0.52011254 -4.1932342   oNH
11   11 Männlich  65  0.61527778     6.285714 49.2857143 -0.54542056 -0.2218649   oHI
12   12 Männlich  32  0.61190476     4.142857  3.5714286  0.99361952 -5.5490370   yNH
13   13 Weiblich  53  0.66825397     3.857143         NA  0.11981292 -5.0806874   oHI
14   14 Weiblich  26  0.74642857     3.142857  5.7142857  0.53451016 -4.6871461   yNH
15   15 Männlich  79  0.53373016     1.142857 22.1428571  0.15342119 -3.9919238   oHI
16   16 Weiblich  74  0.54424603     1.714286 20.0000000  0.41243931 -4.8463992   oHI
17   17 Weiblich  23  0.78968254     1.285714  4.2857143 -0.29709463 -5.4481262   yNH
18   18 Weiblich  71  0.44880952     4.142857 20.0000000  0.19767077 -4.7527286   oHI
19   19 Weiblich  62  0.85317460     3.428571  9.2857143  0.50257381 -5.0488852   oNH
20   20 Männlich  74  0.66547619     5.428571 12.1428571  0.06610275 -4.4914048   oNH
21   21 Männlich  70  0.82380952     3.857143 28.5714286  0.18928960 -4.7039284   oHI
22   22 Männlich  72  0.54900790     2.714286 24.2857143 -0.42039748 -3.8346510   oHI
23   23 Männlich  70  0.65079365     3.142857 13.5714286  0.67584171 -3.7261235   oNH
24   24 Männlich  66  0.54404762     1.000000 42.1428571 -0.22389045 -0.3050729   oHI
25   25 Weiblich  61  0.43750000     3.428571  2.1428571 -0.05697894 -4.1996750   oNH
26   26 Weiblich  74  0.55694444     4.571429 34.2857143  0.43165971 -3.7906007   oHI
27   27 Männlich  20  0.94047619     5.571429  4.2857143  0.56043053 -5.2032493   yNH
28   28 Männlich  28          NA           NA  4.2857143  0.03071840 -5.0407860   yNH
29   29 Männlich  52  0.21746032     1.000000 30.7142857 -0.14759629 -3.9670805   oHI
30   30 Männlich  26  0.21646825     4.000000 32.1428571  0.05245868 -3.6776272   yHI
31   31 Weiblich  54  0.60436508     1.000000 32.1428571 -0.01393779 -3.9286837   oHI
32   32 Männlich  25  0.57539683     2.571429 12.8571429  0.60496998 -4.9858256   yNH
33   33 Weiblich  21  0.97619048     5.000000  3.5714286  0.98079262 -5.4728019   yNH
34   34 Männlich  22  0.80000000     4.857143  1.4285714  0.35718582 -5.9707787   yNH
35   35 Weiblich  55  0.75515873     3.857143 12.1428571  0.73147408 -4.7604876   oNH
36   36 Weiblich  66  0.54007937     4.000000 15.0000000  0.07910598 -5.2093695   oNH
37   37 Weiblich  19  0.63134921     1.142857  1.4285714  0.41283249 -5.1488176   yNH
38   38 Weiblich  25  0.88928571     3.571429 -3.5714286  0.50956657 -5.4712093   yNH
39   39 Weiblich  69  0.66507940     1.857143 40.7142800 -0.15191368 -1.8982144   oHI
40   40 Männlich  25  0.68253968     2.857143 -1.8181818  0.52105047 -7.6565034   yNH
41   41 Männlich  67  0.73055556     6.142857 24.2857143  1.00322058 -4.6406269   oHI
42   42 Männlich  69  0.86309524     5.142857 15.7142857  0.17456630 -4.3503922   oNH
43   43 Männlich  64  0.55793651     1.285714 42.8571429 -0.15051806 -2.9651632   oHI
44   44 Männlich  59  0.77619048     1.714286 13.5714286  0.25608092 -4.5548983   oNH
45   45 Männlich  26  0.81349206     6.285714  5.7142857  0.74052316 -5.2469595   yNH
46   46 Männlich  70  0.92222222     4.857143 15.7142857  0.47961305 -4.6159467   oNH
47   47 Weiblich  24  0.94047619     3.285714  8.5714286  0.25537513 -6.0563171   yNH
48   48 Weiblich  52  0.91865079     5.285714 17.1428571  0.32072435 -4.5420422   oNH
49   49 Männlich  62  0.57182540     1.285714  8.5714286 -0.37626718 -5.6570704   oNH
50   50 Weiblich  58  0.93253968     3.428571  0.0000000 -0.31622942 -4.5789486   oNH
51   51 Männlich  70  0.78412698     4.000000 36.4285714  0.11742064 -2.2818017   oHI
52   52 Weiblich  68  0.50376984     2.000000 25.7142857          NA  4.1974124   oHI
53   53 Weiblich  67  0.45000000     2.000000  5.0000000 -0.31028452 -5.5908288   oNH
54   54 Weiblich  77  0.68055556     4.571429 20.7142857  0.07919706 -4.5592542   oHI
55   55 Weiblich  75          NA           NA  4.2857143  0.28230141 -2.9311993   oNH
56   56 Weiblich  21  0.90079365     6.142857  0.7142857  0.67266567 -6.7287424   yNH
57   57 Weiblich  80  0.53452381     1.571429 35.0000000 -0.23522029  0.3707454   oHI
58   58 Männlich  72  0.48015873     1.714286 27.8571429 -0.16844325 -2.7014611   oHI
59   59 Weiblich  58  0.64404762     4.285714 16.4285714  0.04560261 -4.4846060   oNH
60   60 Weiblich  66  0.51388889     1.000000 36.4285714 -0.14396727 -4.6928181   oHI
61   61 Männlich  62  0.32500000     1.000000 17.8571429 -0.22675520 -2.8992306   oNH
62   62 Weiblich  69  0.38988095     2.285714 15.0000000  0.06182450 -5.3959891   oNH
63   63 Männlich  74  0.87896825     2.571429 16.4285714  0.10808071 -4.4327475   oNH
64   64 Männlich  56  0.56706349     3.571429  7.1428571  0.49965200 -4.9421024   oNH
65   65 Männlich  62          NA           NA 15.7142857  0.46376049 -5.2646241   oNH
66   66 Männlich  59          NA           NA 27.1428571 -0.14576902 -2.7225389   oHI
67   67 Weiblich  59  0.58690476     1.142857 26.4285714  0.31006156 -3.9050845   oHI
68   68 Männlich  73  0.72738095     1.571429 45.0000000 -0.36525244 -2.3651905   oHI
69   69 Männlich  65  0.46984127     1.714286 18.5714286  0.03998730 -4.4431824   oNH
70   70 Männlich  82          NA           NA 32.8571429 -0.17561008 -1.4175314   oHI
71   71 Weiblich  72  0.88888889     2.428571 20.7142857  0.25899414 -4.1778322   oHI
72   72 Männlich  36  0.81250000     5.000000  4.2857143  0.12001266  1.3828984   yNH
73   73 Männlich  25  0.98611111     6.285714  3.5714286  0.21208196 -5.9076965   yNH
74   74 Männlich  67  0.53908730     1.142857 36.4285714 -0.53210637 -2.6406704   oHI
75   75 Männlich  23  0.85158730     3.000000  0.7142857  1.12531720 -6.0031699   yNH
76   76 Männlich  24  0.81349206     2.285714  4.2857143  0.42123250 -5.5145748   yNH
77   77 Männlich  24  1.00000000     1.571429  6.4285714  0.26171182 -5.1276367   yNH
78   78 Weiblich  55  0.65079365     3.142857 17.8571429  0.14092879 -4.0696292   oNH
79   79 Männlich  67  0.66071429     1.000000 31.4285714  0.14135947 -2.3957910   oHI
80   80 Männlich  67  0.63690476     1.571429 21.4285714  0.16019873 -4.0294484   oHI
81   81 Männlich  56  0.09623016     1.857143 28.5714286          NA -3.7164673   oHI
82   82 Weiblich  78  0.33154762     5.571429 23.5714286  0.68414405 -4.2709516   oHI
83   83 Weiblich  64          NA           NA 10.0000000 -0.21946192 -4.8374833   oNH
84   84 Männlich  32  0.95952381     5.285714  3.5714286  0.60682526 -5.6584308   yNH
85   85 Weiblich  63  0.56190476     2.285714  9.2857143  0.18774939 -4.9560247   oNH
86   86 Weiblich  21  0.54186508     4.571429  5.0000000  0.11820314 -5.3485427   yNH
87   87 Weiblich  26  0.81746032     1.428571  4.2857143  0.13908754 -5.7763143   yNH
88   88 Weiblich  57  0.56984127     1.000000 28.5714286          NA         NA   oHI
89   89 Weiblich  71  0.43888889     2.000000 20.7142857  0.69348652 -4.9417715   oHI
90   90 Männlich  25  1.00000000     6.142857  4.2857143  1.10935931 -5.8598283   yNH
91   91 Männlich  27  0.64722222     2.714286  7.8571429  0.32255450 -5.5141401   yNH
92   92 Weiblich  50  0.82142857     3.428571  3.5714286 -0.05113811 -4.8708180   oNH
"

# ------ Convert data ------
# Read the data from the embedded string into a data frame.
export_ind_diff_results <- read.table(text = data_text,
                                      header = TRUE,
                                      stringsAsFactors = FALSE,
                                      na.strings = "NA" # "NA" is converted to missing values.
)
head(export_ind_diff_results)
ind_diff_results <- export_ind_diff_results

# ------ MSA: Complete Ridge Regression ------

#  Data Preparation 
# Select relevant columns and remove missing values.
data_MSA_GMS_training <- ind_diff_results %>% 
  dplyr::select(PTA_full, GMS_training, age, BDS_average, MSA) %>% 
  na.omit() %>% 
  as.data.frame()

# Use the prepared data for modeling.
data.model <- data_MSA_GMS_training

# Standardise the predictors (all variables except the dependent variable "MSA").
data_standard <- data.model %>%
  mutate(across(-MSA, scale))
data_standard %>% count()  # Check count (n = 89 expected)
data_standard <- data_standard %>% na.omit()  # Final n = 82
data <- data_standard  # Final dataset for modeling

# Create the predictor matrix and response vector.
x <- as.matrix(data[, c("PTA_full", "GMS_training", "age", "BDS_average")])
y <- data$MSA

#  Ridge Regression Model Fitting & Bootstrapping 
# Determine the optimal lambda via 10-fold cross-validation (alpha = 0 for ridge).
cv.ridge <- cv.glmnet(x, y, alpha = 0, standardize = TRUE, nfolds = 10)
optimal_lambda <- cv.ridge$lambda.min
cat("Optimal lambda:", optimal_lambda, "\n")

# Fit the final ridge regression model with the optimal lambda.
ridge_model <- glmnet(x, y, alpha = 0, lambda = optimal_lambda, standardize = TRUE)

# Extract coefficients (including the intercept).
ridge_coef <- coef(ridge_model)
observed_coef <- as.vector(ridge_coef)
coef_names <- rownames(ridge_coef)
cat("Observed coefficients:\n")
print(data.frame(Variable = coef_names, Estimate = observed_coef))

# Define a function to refit the ridge model on bootstrap samples.
# 'data' is the full dataset, 'indices' selects the bootstrap sample, and
# 'lambda_value' is the optimal lambda.
ridge_coef_func <- function(data, indices, lambda_value) {
  d <- data[indices, ]
  x_boot <- as.matrix(d[, c("PTA_full", "GMS_training", "age", "BDS_average")])
  y_boot <- d$MSA
  model_boot <- glmnet(x_boot, y_boot, alpha = 0, lambda = lambda_value, standardize = TRUE)
  return(as.vector(coef(model_boot)))
}

# Perform bootstrap resampling with 1000 replications.
nboot <- 1000
boot_results <- boot(data = data, 
                     statistic = function(data, indices) 
                       ridge_coef_func(data, indices, optimal_lambda),
                     R = nboot)

# Bootstrap Confidence Intervals & p-values 
# Initialize vectors to store confidence intervals and p-values.
n_coef <- length(coef_names)
ci_lower <- numeric(n_coef)
ci_upper <- numeric(n_coef)
bootstrap_p_values <- numeric(n_coef)

# Loop over each coefficient to calculate 95% percentile CIs and two-sided p-values.
for (i in 1:n_coef) {
  boot_estimates <- boot_results$t[, i]
  
  # Obtain 95% percentile confidence interval for the i-th coefficient.
  ci <- boot.ci(boot_results, type = "perc", index = i)
  ci_lower[i] <- ci$perc[4]
  ci_upper[i] <- ci$perc[5]
  
  # Compute two-sided bootstrap p-value.
  if (observed_coef[i] > 0) {
    p_val <- 2 * mean(boot_estimates <= 0)
  } else if (observed_coef[i] < 0) {
    p_val <- 2 * mean(boot_estimates >= 0)
  } else {
    p_val <- 1.0
  }
  bootstrap_p_values[i] <- min(p_val, 1)
}

# Create a summary table with coefficients, CIs, and p-values.
results_table <- data.frame(
  Variable = coef_names,
  Estimate = round(observed_coef, 4),
  CI_lower = round(ci_lower, 4),
  CI_upper = round(ci_upper, 4),
  p_value = round(bootstrap_p_values, 3)
)
cat("\nBootstrap Inference Results:\n")
print(results_table)

# Prepare Data for ggplot Density Plot 
# We will plot the bootstrap distributions for the four predictors (excluding the intercept).
desired_order <- c("age", "GMS_training", "BDS_average", "PTA_full")

# Extract bootstrap replicates and set appropriate column names.
boot_mat <- as.data.frame(boot_results$t)
colnames(boot_mat) <- coef_names
boot_plot_data <- boot_mat %>% select(PTA_full, BDS_average, GMS_training, age)

# Reshape data to long format for ggplot.
boot_long <- melt(boot_plot_data, variable.name = "Variable", value.name = "Coefficient")
boot_long$Variable <- factor(boot_long$Variable, levels = desired_order)

# Prepare annotation data from the results table (excluding the intercept).
results_table_plot <- results_table %>% filter(Variable != "(Intercept)")
results_table_plot$Variable <- factor(results_table_plot$Variable, levels = desired_order)
results_table_plot <- results_table_plot %>% mutate(y_pos = as.numeric(Variable))

# Define Custom Colors for Plot 
custom_colors <- c("PTA_full" = "black",
                   "BDS_average" = "#00008B",
                   "GMS_training" = "#A50000",
                   "age" = "#188F60")

# Create ggplot Density (Ridgeline) Plot with Annotations 
density_plot <- ggplot(boot_long, aes(x = Coefficient, y = Variable, fill = Variable)) +
  geom_density_ridges(scale = 0.7, alpha = 0.8, color = "white", size = 0.1) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +  # Reference line at 0
  geom_segment(data = results_table_plot,
               aes(x = CI_lower, xend = CI_upper, y = y_pos, yend = y_pos),
               color = "black", size = 1) +  # 95% CI horizontal lines
  geom_segment(data = results_table_plot,
               aes(x = Estimate, xend = Estimate, 
                   y = y_pos - 0.15, yend = y_pos + 0.15),
               color = "black", size = 1) +  # Observed coefficient ticks
  geom_segment(data = results_table_plot,
               aes(x = CI_lower, xend = CI_lower, 
                   y = y_pos - 0.07, yend = y_pos + 0.07),
               color = "black", size = 1) +  # CI boundary ticks
  geom_segment(data = results_table_plot,
               aes(x = CI_upper, xend = CI_upper, 
                   y = y_pos - 0.07, yend = y_pos + 0.07),
               color = "black", size = 1) +
  scale_fill_manual(values = custom_colors) +
  scale_y_discrete(limits = desired_order) +
  labs(title = "Bootstrap Distributions of Ridge Regression Coefficients",
       x = "Coefficient Value", y = "Predictor") +
  theme_bw(base_size = 20) +
  theme(axis.text = element_text(size = 18),
        axis.title = element_text(size = 20, face = "bold"),
        legend.position = "none")

# Display and save the density plot.
print(density_plot)

# Final Reporting 
cat("\n--- FINAL REPORTING ---\n\n")
cat("1. Ridge Regression (Bootstrap) Inference Results:\n")
print(results_table)


# ------ SRT: Complete Ridge Regression ------
# (Procedures for SRT mirror those for MSA; minimal comments provided.)

SRT_data_standard <- ind_diff_results_inv %>% 
  dplyr::select(PTA_full, GMS_training, age, BDS_average, SRT) %>% 
  na.omit() %>% 
  as.data.frame()

nrow(SRT_data_standard)
data <- SRT_data_standard

# Provide essential summary statistics.
SRT_data_standard %>%
  summarise_all(~ list(
    mean = mean(.),
    sd = sd(.),
    min = min(.),
    max = max(.),
    median = median(.)
  )) %>%
  unnest(cols = everything())

# Standardise predictors (excluding SRT).
SRT_data_standard <- SRT_data_standard %>%
  mutate(across(-SRT, scale))
data <- SRT_data_standard

# Create predictor matrix and response vector.
x <- as.matrix(data[, c("PTA_full", "GMS_training", "age", "BDS_average")])
y <- data$SRT

# Determine optimal lambda via 10-fold cross-validation.
cv.ridge <- cv.glmnet(x, y, alpha = 0, standardize = TRUE, nfolds = 10)
optimal_lambda <- cv.ridge$lambda.min
cat("Optimal lambda:", optimal_lambda, "\n")

# Fit final ridge regression model.
ridge_model <- glmnet(x, y, alpha = 0, lambda = optimal_lambda, standardize = TRUE)
ridge_coef <- coef(ridge_model)
observed_coef <- as.vector(ridge_coef)
coef_names <- rownames(ridge_coef)
cat("Observed coefficients:\n")
print(data.frame(Variable = coef_names, Estimate = observed_coef))

# Define function to refit ridge model on bootstrap samples.
ridge_coef_func <- function(data, indices, lambda_value) {
  d <- data[indices, ]
  x_boot <- as.matrix(d[, c("PTA_full", "GMS_training", "age", "BDS_average")])
  y_boot <- d$SRT
  model_boot <- glmnet(x_boot, y_boot, alpha = 0, lambda = lambda_value, standardize = TRUE)
  return(as.vector(coef(model_boot)))
}

# Perform bootstrap with 1000 replications.
nboot <- 1000
set.seed(123)
boot_results <- boot(data = data, 
                     statistic = function(data, indices) 
                       ridge_coef_func(data, indices, optimal_lambda),
                     R = nboot)

# Bootstrap confidence intervals and p-values.
n_coef <- length(coef_names)
ci_lower <- numeric(n_coef)
ci_upper <- numeric(n_coef)
bootstrap_p_values <- numeric(n_coef)

for (i in 1:n_coef) {
  boot_estimates <- boot_results$t[, i]
  ci <- boot.ci(boot_results, type = "perc", index = i)
  ci_lower[i] <- ci$perc[4]
  ci_upper[i] <- ci$perc[5]
  
  if (observed_coef[i] > 0) {
    p_val <- 2 * mean(boot_estimates <= 0)
  } else if (observed_coef[i] < 0) {
    p_val <- 2 * mean(boot_estimates >= 0)
  } else {
    p_val <- 1.0
  }
  bootstrap_p_values[i] <- min(p_val, 1)
}

results_table <- data.frame(
  Variable = coef_names,
  Estimate = round(observed_coef, 4),
  CI_lower = round(ci_lower, 4),
  CI_upper = round(ci_upper, 4),
  p_value = round(bootstrap_p_values, 3)
)
cat("\nBootstrap Inference Results:\n")
print(results_table)

# Prepare data for ggplot density plot.
desired_order <- c("age", "GMS_training", "BDS_average", "PTA_full")
boot_mat <- as.data.frame(boot_results$t)
colnames(boot_mat) <- coef_names
boot_plot_data <- boot_mat %>% select(PTA_full, BDS_average, GMS_training, age)
boot_long <- melt(boot_plot_data, variable.name = "Variable", value.name = "Coefficient")
boot_long$Variable <- factor(boot_long$Variable, levels = desired_order)
results_table_plot <- results_table %>% filter(Variable != "(Intercept)")
results_table_plot$Variable <- factor(results_table_plot$Variable, levels = desired_order)
results_table_plot <- results_table_plot %>% mutate(y_pos = as.numeric(Variable))

custom_colors <- c("PTA_full" = "black",
                   "BDS_average" = "#00008B",
                   "GMS_training" = "#A50000",
                   "age" = "#188F60")

density_plot <- ggplot(boot_long, aes(x = Coefficient, y = Variable, fill = Variable)) +
  geom_density_ridges(scale = 0.7, alpha = 0.8, color = "white", size = 0.1) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_segment(data = results_table_plot,
               aes(x = CI_lower, xend = CI_upper, y = y_pos, yend = y_pos),
               color = "black", size = 1) +
  geom_segment(data = results_table_plot,
               aes(x = Estimate, xend = Estimate, 
                   y = y_pos - 0.15, yend = y_pos + 0.15),
               color = "black", size = 1) +
  geom_segment(data = results_table_plot,
               aes(x = CI_lower, xend = CI_lower, 
                   y = y_pos - 0.07, yend = y_pos + 0.07),
               color = "black", size = 1) +
  geom_segment(data = results_table_plot,
               aes(x = CI_upper, xend = CI_upper, 
                   y = y_pos - 0.07, yend = y_pos + 0.07),
               color = "black", size = 1) +
  scale_fill_manual(values = custom_colors) +
  scale_y_discrete(limits = desired_order) +
  labs(title = "Bootstrap Distributions of Ridge Regression Coefficients",
       x = "Coefficient Value", y = "Predictor") +
  theme_bw(base_size = 20) +
  theme(axis.text = element_text(size = 18),
        axis.title = element_text(size = 20, face = "bold"),
        legend.position = "none")

print(density_plot)
ggsave("pub_plots/multifigure/Ridge_density-plot_SRT.png", density_plot, width = 8, height = 7.3, dpi = 320)

cat("\n--- FINAL REPORTING ---\n\n")
cat("1. Ridge Regression (Bootstrap) Inference Results:\n")
print(results_table)









































# -------------- XGBoost analysis (model selection loop) -----------------------

# ------ Data cleaning ------
# Select only the relevant variables for the MSA analysis.
data <- ind_diff_results %>% dplyr::select(PTA_full, GMS_training, age, BDS_average, MSA)
# Remove rows with missing values in the MSA variable.
data <- data[complete.cases(data$MSA), ]
data <- as.data.frame(data)

# ------ Feature selection ------
features <- c("PTA_full", "GMS_training", "age", "BDS_average")
target <- "MSA"

# ------ Hyperparameter bounds ------
bounds <- list(
  eta = c(0.05, 0.15),
  max_depth = c(3L, 5L),
  subsample = c(0.5, .9),
  colsample_bytree = c(0.65, .75),
  min_child_weight = c(8,16),
  lambda = c(0.3, 1),
  alpha = c(0.3, 1),
  gamma = c(0, .5)
)

# ------ Generate random config function ------
# This function samples a random set of hyperparameters from the given bounds.
generate_random_config <- function(bounds) {
  list(
    eta = runif(1, bounds$eta[1], bounds$eta[2]),
    max_depth = sample(bounds$max_depth[1]:bounds$max_depth[2], 1),
    subsample = runif(1, bounds$subsample[1], bounds$subsample[2]),
    colsample_bytree = runif(1, bounds$colsample_bytree[1], bounds$colsample_bytree[2]),
    min_child_weight = runif(1, bounds$min_child_weight[1], bounds$min_child_weight[2]),
    lambda = runif(1, bounds$lambda[1], bounds$lambda[2]),
    alpha = runif(1, bounds$alpha[1], bounds$alpha[2]),
    gamma = runif(1, bounds$gamma[1], bounds$gamma[2])
  )
}
# Test the function (should output a list of hyperparameters)
generate_random_config(bounds) # works perfectly

# ------ XGBoost run #1: MSA ------
num_runs <- 2000         # Number of hyperparameter configurations to compare
n_iterations <- 500      # Number of repetitions to average performance

# Initialize results storage (ensure 'all_model_results' is defined)
all_model_results <- tibble()

for(var in 1:num_runs) {
  
  # Generate a random hyperparameter configuration for xgboost.
  best_model <- generate_random_config(bounds) 
  
  # Randomly choose the number of boosting rounds (trees) between 60 and 100.
  nround <- round(runif(1, 60, 100), digits = 0)
  
  # Randomly select the training split (85% to 90% of the data for training).
  split <- round(runif(1, .85, .9), digits = 2)
  
  # Initialize matrices for storing performance metrics over iterations.
  train.MAE <- matrix(NA, nrow = n_iterations, ncol = 1)
  train.RSME <- matrix(NA, nrow = n_iterations, ncol = 1)
  train.Rsquared_carret <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.MAE <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.RSME <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.Rsquared_carret <- matrix(NA, nrow = n_iterations, ncol = 1)
  
  # Lists to store training and test data along with predictions.
  train_data_list <- list()
  test_data_list <- list()
  train_predictions_list <- list()
  test_predictions_list <- list()
  
  # Remove any existing random seed for fresh randomness.
  rm(list=".Random.seed", envir=globalenv())
  
  # Loop over iterations to evaluate the model performance variability.
  for (i in 1:n_iterations) {
    # Create a partition that attempts to balance the target (MSA) distribution.
    indices <- createDataPartition(data$MSA, p = split, list = FALSE)
    train_data_iter <- data[indices,]
    test_data_iter <- data[-indices,]
    
    # Convert training features and target variable for xgboost.
    train_features_iter <- as.matrix(train_data_iter[,features])
    train_target_iter <- as.matrix(train_data_iter[,target])
    
    # Train the xgboost model:
    # - Use all columns except the 5th (assumed to be the target) as predictors.
    # - Use column 5 as the label.
    # - Apply the random hyperparameters (best_model).
    # - Use nround boosting rounds.
    # - Suppress output with verbose = 0.
    model.inspect <- xgboost(
      data = as.matrix(train_data_iter[, -5]),
      label = train_data_iter[, 5],
      params = best_model,
      nround = nround,
      verbose = 0
    )
    
    # Compute performance metrics on the training set.
    train.MAE[i,] <- caret::MAE(train_data_iter$MSA, predict(model.inspect, as.matrix(train_data_iter[,-5]))) 
    train.RSME[i,] <- caret::RMSE(train_data_iter$MSA, predict(model.inspect, as.matrix(train_data_iter[,-5]))) 
    train.Rsquared_carret[i,] <- caret::R2(predict(model.inspect, as.matrix(train_data_iter[, -5])), train_data_iter$MSA)
    
    # Compute performance metrics on the test set.
    test.MAE[i,] <- caret::MAE(test_data_iter$MSA, predict(model.inspect, as.matrix(test_data_iter[,-5]))) 
    test.RSME[i,] <- caret::RMSE(test_data_iter$MSA, predict(model.inspect, as.matrix(test_data_iter[,-5]))) 
    test.Rsquared_carret[i,] <- caret::R2(predict(model.inspect, as.matrix(test_data_iter[, -5])), test_data_iter$MSA)
    
    # Generate predictions for both training and test sets.
    train_preds <- predict(model.inspect, as.matrix(train_data_iter[,-5]))
    test_preds <- predict(model.inspect, as.matrix(test_data_iter[,-5]))
    
    # Store data and predictions for later analysis.
    train_data_list[[i]] <- cbind(train_data_iter, Predicted_MSA = train_preds)
    test_data_list[[i]] <- cbind(test_data_iter, Predicted_MSA = test_preds)
    
    train_predictions_list[[i]] <- train_preds
    test_predictions_list[[i]] <- test_preds
  }
  
  # Initialize a data frame to store Pearson correlation values for each iteration.
  correlation_results <- data.frame(Iteration = integer(n_iterations), 
                                    Train_Correlation = numeric(n_iterations), 
                                    Test_Correlation = numeric(n_iterations))
  
  # Calculate Pearson correlation between actual and predicted MSA values.
  for (i in 1:n_iterations) {
    train_actual <- train_data_list[[i]]$MSA
    train_predicted <- train_data_list[[i]]$Predicted_MSA
    test_actual <- test_data_list[[i]]$MSA
    test_predicted <- test_data_list[[i]]$Predicted_MSA
    train_corr <- cor(train_actual, train_predicted, method = "pearson")
    test_corr <- cor(test_actual, test_predicted, method = "pearson")
    
    correlation_results[i, "Iteration"] <- i
    correlation_results[i, "Train_Correlation"] <- train_corr
    correlation_results[i, "Test_Correlation"] <- test_corr
  }
  
  # Helper function to combine configuration and performance metrics into a tibble.
  store_model_results <- function(config, split, nround, train_results, test_results, correlation_results) {
    tibble(
      eta = config$eta,
      max_depth = config$max_depth,
      subsample = config$subsample,
      colsample_bytree = config$colsample_bytree,
      min_child_weight = config$min_child_weight,
      lambda = config$lambda,
      alpha = config$alpha,
      gamma = config$gamma,
      split = split,
      nround = nround, 
      train_MAE = train_results$MAE,
      test_MAE = test_results$MAE,
      train_RMSE = train_results$RMSE,
      test_RMSE = test_results$RMSE,
      train_Rsquared_carret = train_results$Rsquared_carret,
      test_Rsquared_carret = test_results$Rsquared_carret,
      train_correlation = correlation_results$Train_Correlation,
      test_correlation = correlation_results$Test_Correlation
    )
  }
  
  # Compute average performance metrics over all iterations.
  train_results <- list(
    MAE = mean(train.MAE),
    RMSE = mean(train.RSME),
    Rsquared_carret = mean(train.Rsquared_carret)
  )
  
  test_results <- list(
    MAE = mean(test.MAE),
    RMSE = mean(test.RSME),
    Rsquared_carret = mean(test.Rsquared_carret)
  )
  
  correlation_results <- list(
    Train_Correlation = mean(correlation_results$Train_Correlation),
    Test_Correlation = mean(correlation_results$Test_Correlation)
  )
  
  # Store the results from this hyperparameter configuration run.
  model_results <- store_model_results(
    config = best_model,
    split = split,
    nround = nround,
    train_results = train_results,
    test_results = test_results,
    correlation_results = correlation_results
  )
  
  # Append current run's results to overall results.
  all_model_results <- bind_rows(all_model_results, model_results)
  
  # Optionally display the best configuration for debugging.
  all_model_results %>% 
    select(eta, max_depth, subsample, colsample_bytree, min_child_weight, lambda,
           alpha, gamma, split, nround, train_Rsquared_carret, test_Rsquared_carret)  %>%
    arrange(desc(test_Rsquared_carret)) %>% head(1)
  print(num_runs)
} 

# save results for redundancy
all_model_results_save <- all_model_results

# Display combined results, ordering by the difference between train and test R^2.
all_model_results %>% 
  select(eta, max_depth, subsample, colsample_bytree, min_child_weight, lambda,
         alpha, gamma, split, nround, train_Rsquared_carret, test_Rsquared_carret) %>%
  mutate(r2_diff = train_Rsquared_carret - test_Rsquared_carret) %>% 
  arrange(r2_diff, desc(test_Rsquared_carret))


# adapt data frame
all_model_results <- all_model_results %>% mutate(diff_RMSE = train_RMSE - test_RMSE,
                             diff_R2 = train_Rsquared_carret - test_Rsquared_carret) 

all_model_results <- all_model_results %>% mutate(row_nr2 = 1:nrow(tmp.data_random21316)) 

# try another loop for SHAP values
df_loop.new <- tmp.data_random21316 %>% 
  filter(diff_RMSE < .04) %>% # these are the bounds that lead to the top20 within the paper  
  filter(diff_R2 < .04) %>%
  filter(test_Rsquared_carret > .30)

# just to double check initialize the data again
features <- c("PTA_full", "GMS_training", "age", "BDS_average")
target <- "MSA"

# h8z
data_MSA_GMS_training <- ind_diff_results %>% dplyr::select(PTA_full, GMS_training, age, BDS_average, MSA)

# omit NA only in MSA coloumn
data_MSA_GMS_training <- data_MSA_GMS_training %>% filter(!is.na(MSA)) %>% as.data.frame()

# which data to use?
data.model <- data_MSA_GMS_training # versus data_MSA_GMSI

# clear any .Random.seed
rm(list=".Random.seed", envir=globalenv())

# The top 20 configurations are extracted into df_top20.correct.
df_top20.correct <- df_loop.new[1:20, ]

# (Optional: if needed, update all_model_results with additional metrics here.)
# For SHAP value calculation, only the existing configuration parameters are required.

# Reinitialize essential variables for SHAP calculation.
features <- c("PTA_full", "GMS_training", "age", "BDS_average")
target <- "MSA"

# Load the MSA data and remove rows with missing MSA values.
data_MSA_GMS_training <- ind_diff_results %>% 
  dplyr::select(PTA_full, GMS_training, age, BDS_average, MSA) %>% 
  filter(!is.na(MSA)) %>% 
  as.data.frame()

# Use this dataset for model training.
data.model <- data_MSA_GMS_training

# Clear any existing random seed.
rm(list = ".Random.seed", envir = globalenv())

# ------ Calculate SHAP Values for Top 20 Configurations ------

# Initialize a list to store SHAP results for each configuration.
all_shap_results <- list()

# Loop over each configuration (each row in df_top20.correct).
for(i in seq_len(nrow(df_top20.correct))) {
  
  # Extract current configuration parameters.
  curr_row <- df_top20.correct[i, ]
  
  current_model_params <- list(
    eta              = curr_row[["eta"]], 
    max_depth        = curr_row[["max_depth"]],
    subsample        = curr_row[["subsample"]],
    colsample_bytree = curr_row[["colsample_bytree"]],
    min_child_weight = curr_row[["min_child_weight"]],
    lambda           = curr_row[["lambda"]],
    alpha            = curr_row[["alpha"]],
    gamma            = curr_row[["gamma"]]
  )
  
  # Extract model settings from the current configuration.
  bestmodel_split_current   <- curr_row[["split"]]
  bestmodel_nrounds_current <- curr_row[["nround"]]
  
  # Fit the xgboost model on the full dataset using the current configuration.
  shap_model <- xgboost::xgboost(
    data = as.matrix(data.model[, -5]),
    label = data.model[, 5],
    params = current_model_params,
    nround = bestmodel_nrounds_current,
    verbose = 0
  )
  
  # Calculate SHAP values using kernelshap and shapviz on the full dataset.
  shap_s <- kernelshap::kernelshap(shap_model, 
                                   X = as.matrix(data.model[, -5]), 
                                   bg_X = as.matrix(data.model[, -5]))
  shap_sv <- shapviz::shapviz(shap_s)
  
  # Extract SHAP values and rename columns to reflect corresponding predictors.
  tmp_SHAP <- shap_sv$S %>% 
    as.data.frame() %>% 
    dplyr::rename(
      SHAP_BDS = BDS_average,
      SHAP_GMS = GMS_training,
      SHAP_AGE = age,
      SHAP_PTA = PTA_full
    )
  
  # Calculate the average absolute SHAP values for each predictor across all observations.
  avg_SHAP <- colMeans(abs(tmp_SHAP[, c("SHAP_PTA", "SHAP_GMS", "SHAP_AGE", "SHAP_BDS")]), na.rm = TRUE)
  
  # Compute relative SHAP importance (proportion of the total absolute SHAP).
  total_abs <- sum(avg_SHAP)
  rel_SHAP <- avg_SHAP / total_abs
  
  # Create a data frame containing the calculated SHAP values.
  shap_df <- data.frame(
    SHAP_PTA     = avg_SHAP["SHAP_PTA"],
    SHAP_GMS     = avg_SHAP["SHAP_GMS"],
    SHAP_AGE     = avg_SHAP["SHAP_AGE"],
    SHAP_BDS     = avg_SHAP["SHAP_BDS"]
  )
  
  # Store the SHAP values using an identifier based on the row index.
  key_name <- paste0("Row_", i)
  all_shap_results[[key_name]] <- shap_df
}

# Combine SHAP results from all configurations into one data frame.
shap_results_combined <- dplyr::bind_rows(all_shap_results, .id = "Row_ID")

# ------ Merge SHAP Values with Top 20 Configurations ------

# Add a Row_ID column to df_top20.correct to match the identifiers used above.
df_top20.correct <- df_top20.correct %>% 
  mutate(Row_ID = paste0("Row_", seq_len(nrow(df_top20.correct))))

# Merge the original top 20 configurations with the SHAP values.
df_top20_with_SHAP <- merge(df_top20.correct, shap_results_combined, by = "Row_ID", all.x = TRUE)




# -------------- PLOT SHAP DATA -------------------------

df_smooth_decide <- 8  # Smoothing parameter

# Define Function for SHAP Plot 
plot_shap_var <- function(df, base_var, shap_var, x_label, y_label, plot_title, 
                          x_breaks, x_limits, errorbar_width, color_values, shape_values, 
                          y_breaks, y_limits) {
  # Compute summary statistics by p_id and group.
  df_summary <- df %>%
    group_by(p_id, group) %>%
    summarise(
      x_mean = first(!!sym(base_var)),
      shap_mean = mean(!!sym(shap_var), na.rm = TRUE),
      shap_sd = sd(!!sym(shap_var), na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(
      shap_se = shap_sd / sqrt(n),
      t_crit = qt(0.995, df = n - 1),  # 99% interval (use 0.975 for 95%)
      shap_ci_low = shap_mean - t_crit * shap_se,
      shap_ci_high = shap_mean + t_crit * shap_se
    )
  
  # Bin raw data by the base variable.
  num_bins <- 60
  df_binned <- df %>%
    mutate(x_bin = cut(!!sym(base_var), breaks = num_bins, include.lowest = TRUE)) %>%
    group_by(x_bin) %>%
    summarise(
      x_mid = mean(!!sym(base_var), na.rm = TRUE),
      y_2.5 = quantile(!!sym(shap_var), probs = 0.025, na.rm = TRUE),
      y_10 = quantile(!!sym(shap_var), probs = 0.10, na.rm = TRUE),
      y_25 = quantile(!!sym(shap_var), probs = 0.25, na.rm = TRUE),
      y_50 = quantile(!!sym(shap_var), probs = 0.50, na.rm = TRUE),
      y_75 = quantile(!!sym(shap_var), probs = 0.75, na.rm = TRUE),
      y_90 = quantile(!!sym(shap_var), probs = 0.90, na.rm = TRUE),
      y_97.5 = quantile(!!sym(shap_var), probs = 0.975, na.rm = TRUE),
      .groups = "drop"
    ) %>% arrange(x_mid)
  
  # Smooth the percentile curves using cubic splines.
  df_binned_smooth <- df_binned %>%
    mutate(
      y_2.5_smooth  = as.numeric(smooth.spline(x_mid, y_2.5, df = df_smooth_decide)$y),
      y_97.5_smooth = as.numeric(smooth.spline(x_mid, y_97.5, df = df_smooth_decide)$y),
      y_10_smooth   = as.numeric(smooth.spline(x_mid, y_10, df = df_smooth_decide)$y),
      y_90_smooth   = as.numeric(smooth.spline(x_mid, y_90, df = df_smooth_decide)$y),
      y_25_smooth   = as.numeric(smooth.spline(x_mid, y_25, df = df_smooth_decide)$y),
      y_75_smooth   = as.numeric(smooth.spline(x_mid, y_75, df = df_smooth_decide)$y)
    )
  
  # Create the SHAP plot with nested coverage ribbons and summary error bars.
  p <- ggplot() +
    geom_ribbon(data = df_binned_smooth, 
                aes(x = x_mid, ymin = y_2.5_smooth, ymax = y_97.5_smooth), 
                fill = "grey60", alpha = 0.2) +
    geom_ribbon(data = df_binned_smooth, 
                aes(x = x_mid, ymin = y_10_smooth, ymax = y_90_smooth), 
                fill = "grey50", alpha = 0.2) +
    geom_ribbon(data = df_binned_smooth, 
                aes(x = x_mid, ymin = y_25_smooth, ymax = y_75_smooth), 
                fill = "grey40", alpha = 0.2) +
    geom_errorbar(data = df_summary, 
                  aes(x = x_mean, ymin = shap_ci_low, ymax = shap_ci_high, color = group), 
                  width = errorbar_width, alpha = 0.7) +
    geom_point(data = df_summary, 
               aes(x = x_mean, y = shap_mean, color = group, shape = group), 
               alpha = 0.9, size = 4) +
    labs(title = plot_title, x = x_label, y = y_label) +
    theme_bw(base_size = 20) +
    theme(axis.text = element_text(size = 18),
          axis.title = element_text(size = 20, face = "bold"),
          legend.position = "none") +
    scale_color_manual(values = color_values) +
    scale_shape_manual(values = shape_values) +
    scale_y_continuous(breaks = y_breaks, limits = y_limits) +
    scale_x_continuous(breaks = x_breaks, limits = x_limits)
  
  return(p)
}

# ------ Prepare Data for SHAP Plots --------------
# 'df' is assumed to come from a previous processing step.
# Here we start with final_combined_results_top20correct and add required columns.
df <- df_top20_with_SHAP
# Create a group variable based on PTA_full and age.
df <- df %>% mutate(group = case_when(
  (PTA_full >= 20 & age < 50) ~ "yHI",
  (PTA_full >= 20 & age > 49) ~ "oHI",
  (PTA_full < 20 & age < 50)  ~ "yNH",
  (PTA_full < 20 & age > 49)  ~ "oNH"
))
df$group <- factor(df$group, levels = c("yNH", "oNH", "yHI", "oHI"))

# Invert SHAP values.
df <- df %>% mutate(across(starts_with("SHAP_"), ~ . * -1)) # optional! just to align the Plot visually with the SRT plot.

# ------ Common Aesthetics --------------
color_vals <- c("yNH" = "#0073c2", "oNH" = "#efc000", "oHI" = "#868686", "yHI" = "#cd534c")
shape_vals <- c("yNH" = 16, "oNH" = 17, "oHI" = 15, "yHI" = 3)

# ------ SHAP Plot for BDS --------------
p_BDS <- plot_shap_var(
  df = df,
  base_var = "BDS_average",
  shap_var = "SHAP_BDS",
  x_label = "BDS",
  y_label = "MSA - SHAP BDS",
  plot_title = "MSA: BDS / WMC",
  x_breaks = seq(0.1, 1, 0.1),
  x_limits = c(0.1, 1.01),
  errorbar_width = 0.01,
  color_values = color_vals,
  shape_values = shape_vals,
  y_breaks = c(-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.10, 0.15),
  y_limits = c(-0.115, 0.15)
)
print(p_BDS)
# ggsave("pub plots/multifigure/SHAP_BDS_training_error.png", p_BDS, width = 8, height = 8, dpi = 320)

# ------ SHAP Plot for PTA --------------
p_PTA <- plot_shap_var(
  df = df,
  base_var = "PTA_full",
  shap_var = "SHAP_PTA",
  x_label = "PTA",
  y_label = "MSA - SHAP PTA",
  plot_title = "MSA - PTA",
  x_breaks = seq(-10, 50, 10),
  x_limits = c(-10, 50.01),
  errorbar_width = 0.5,
  color_values = color_vals,
  shape_values = shape_vals,
  y_breaks = c(-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.10, 0.15),
  y_limits = c(-0.115, 0.15)
)
print(p_PTA)
# ggsave("pub plots/multifigure/SHAP_MSA_PTA_error.png", p_PTA, width = 8, height = 8, dpi = 320)

# ------ SHAP Plot for GMS --------------
p_GMS <- plot_shap_var(
  df = df,
  base_var = "GMS_training",
  shap_var = "SHAP_GMS",
  x_label = "GMS",
  y_label = "MSA - SHAP GMS",
  plot_title = "MSA - GMS",
  x_breaks = seq(1, 7, 1),
  x_limits = c(0.99, 7.01),
  errorbar_width = 0.055,
  color_values = color_vals,
  shape_values = shape_vals,
  y_breaks = c(-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.10, 0.15),
  y_limits = c(-0.115, 0.15)
)
print(p_GMS)
# ggsave("pub plots/multifigure/SHAP_MSA_GMS_error.png", p_GMS, width = 8, height = 8, dpi = 320)

# ------ SHAP Plot for AGE --------------
p_AGE <- plot_shap_var(
  df = df,
  base_var = "age",
  shap_var = "SHAP_AGE",
  x_label = "AGE",
  y_label = "MSA - SHAP AGE",
  plot_title = "MSA - AGE",
  x_breaks = seq(20, 80, 10),
  x_limits = c(17.95, 82.01),
  errorbar_width = 0.7,
  color_values = color_vals,
  shape_values = shape_vals,
  y_breaks = c(-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.10, 0.15),
  y_limits = c(-0.115, 0.15)
)
print(p_AGE)
# ggsave("pub plots/multifigure/SHAP_MSA_AGE_error.png", p_AGE, width = 8, height = 8, dpi = 320)

# ------ XGBoost run #2: SRT ------
# (Minimal comments provided as the approach is similar to the MSA run.)
subset4 <- ind_diff_results %>%
  dplyr::select(
    SRT,
    BDS_average,
    GMS_training,
    age,
    PTA_full
  )

data4 <- subset4[complete.cases(subset4$SRT), ]
nrow(data4)  # 'N = 91'

target_SRT <- "SRT" 
features_SRT <- c("PTA_full", "GMS_training", "age", "BDS_average")
data_matrix_SRT <- as.matrix(data4[, features_SRT])
label_vector_SRT <- data4[, target_SRT]

all_model_results_SRT <- tibble()

num_runs <- 2000         # Number of configurations to compare
n_iterations <- 500     # Repetitions for averaging

for(var in 1:num_runs) {
  
  best_model <- generate_random_config(bounds)
  nround <- round(runif(1, 50, 100), digits = 0)
  split <- round(runif(1, .5, .9), digits = 2)
  
  train.MAE <- matrix(NA, nrow = n_iterations, ncol = 1)
  train.RSME <- matrix(NA, nrow = n_iterations, ncol = 1)
  train.Rsquared <- matrix(NA, nrow = n_iterations, ncol = 1)
  train.Rsquared_carret <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.MAE <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.RSME <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.Rsquared <- matrix(NA, nrow = n_iterations, ncol = 1)
  test.Rsquared_carret <- matrix(NA, nrow = n_iterations, ncol = 1)
  train_data_list <- list()
  test_data_list <- list()
  train_predictions_list <- list()
  test_predictions_list <- list()
  
  rm(list=".Random.seed", envir=globalenv())
  
  for (i in 1:n_iterations) {
    indices <- createDataPartition(data4$SRT, p = split, list = FALSE)
    train_data_iter <- data4[indices,]
    test_data_iter <- data4[-indices,]
    
    train_features_iter <- as.matrix(train_data_iter[,features])
    train_target_iter <- as.matrix(train_data_iter[,"SRT"])
    
    model.inspect <- xgboost(
      data = as.matrix(train_data_iter[, -1]),
      label = train_data_iter[, 1],
      params = best_model,
      nround = nround,
      verbose = 0
    )
    
    train.MAE[i,] <- caret::MAE(train_data_iter$SRT, predict(model.inspect, as.matrix(train_data_iter[,-1]))) 
    train.RSME[i,] <- caret::RMSE(train_data_iter$SRT, predict(model.inspect, as.matrix(train_data_iter[,-1]))) 
    train.Rsquared[i,] <- 1 - sum((train_data_iter$SRT - predict(model.inspect, as.matrix(train_data_iter[,-1])))^2) / sum((train_data_iter$SRT - mean(train_data_iter$SRT))^2)
    train.Rsquared_carret[i,] <- caret::R2(predict(model.inspect, as.matrix(train_data_iter[, -1])), train_data_iter$SRT)
    
    test.MAE[i,] <- caret::MAE(test_data_iter$SRT, predict(model.inspect, as.matrix(test_data_iter[,-1]))) 
    test.RSME[i,] <- caret::RMSE(test_data_iter$SRT, predict(model.inspect, as.matrix(test_data_iter[,-1]))) 
    test.Rsquared[i,] <- 1 - sum((test_data_iter$SRT - predict(model.inspect, as.matrix(test_data_iter[,-1])))^2) / sum((test_data_iter$SRT - mean(predict(model.inspect, as.matrix(test_data_iter[,-1]))))^2)
    test.Rsquared_carret[i,] <- caret::R2(predict(model.inspect, as.matrix(test_data_iter[, -1])), test_data_iter$SRT)
    
    train_preds <- predict(model.inspect, as.matrix(train_data_iter[,-1]))
    test_preds <- predict(model.inspect, as.matrix(test_data_iter[,-1]))
    
    train_data_list[[i]] <- cbind(train_data_iter, Predicted_SRT = train_preds)
    test_data_list[[i]] <- cbind(test_data_iter, Predicted_SRT = test_preds)
    
    train_predictions_list[[i]] <- train_preds
    test_predictions_list[[i]] <- test_preds
  }
  
  correlation_results <- data.frame(Iteration = integer(n_iterations), Train_Correlation = numeric(n_iterations), Test_Correlation = numeric(n_iterations))
  
  for (i in 1:n_iterations) {
    train_actual <- train_data_list[[i]]$SRT
    train_predicted <- train_data_list[[i]]$Predicted_SRT
    test_actual <- test_data_list[[i]]$SRT
    test_predicted <- test_data_list[[i]]$Predicted_SRT
    train_corr <- cor(train_actual, train_predicted, method = "pearson")
    test_corr <- cor(test_actual, test_predicted, method = "pearson")
    
    correlation_results[i, "Iteration"] <- i
    correlation_results[i, "Train_Correlation"] <- train_corr
    correlation_results[i, "Test_Correlation"] <- test_corr
  }
  
  store_model_results <- function(config, split, nround, train_results, test_results, correlation_results) {
    tibble(
      eta = config$eta,
      max_depth = config$max_depth,
      subsample = config$subsample,
      colsample_bytree = config$colsample_bytree,
      min_child_weight = config$min_child_weight,
      lambda = config$lambda,
      alpha = config$alpha,
      gamma = config$gamma,
      split = split,
      nround = nround, 
      train_MAE = train_results$MAE,
      test_MAE = test_results$MAE,
      train_RMSE = train_results$RMSE,
      test_RMSE = test_results$RMSE,
      train_Rsquared = train_results$Rsquared,
      test_Rsquared = test_results$Rsquared,
      train_Rsquared_carret = train_results$Rsquared_carret,
      test_Rsquared_carret = test_results$Rsquared_carret,
      train_correlation = correlation_results$Train_Correlation,
      test_correlation = correlation_results$Test_Correlation
    )
  }
  
  train_results <- list(
    MAE = mean(train.MAE),
    RMSE = mean(train.RSME),
    Rsquared = mean(train.Rsquared),
    Rsquared_carret = mean(train.Rsquared_carret)
  )
  
  test_results <- list(
    MAE = mean(test.MAE),
    RMSE = mean(test.RSME),
    Rsquared = mean(test.Rsquared),
    Rsquared_carret = mean(test.Rsquared_carret)
  )
  
  correlation_results <- list(
    Train_Correlation = mean(correlation_results$Train_Correlation),
    Test_Correlation = mean(correlation_results$Test_Correlation)
  )
  
  model_results <- store_model_results(
    config = best_model,
    split = split,
    nround = nround,
    train_results = train_results,
    test_results = test_results,
    correlation_results = correlation_results
  )
  
  all_model_results_SRT <- bind_rows(all_model_results_SRT, model_results)
  
  all_model_results_SRT %>% 
    select(eta, max_depth, subsample, colsample_bytree, min_child_weight, lambda,
           alpha, gamma, split, nround, train_Rsquared_carret, test_Rsquared_carret)  %>%
    arrange(desc(test_Rsquared_carret)) %>% head(1)
} 


# save results for redundancy
all_model_results_SRT_save <- all_model_results_SRT

# Display combined results, ordering by the difference between train and test R^2.
all_model_results_SRT %>% 
  select(eta, max_depth, subsample, colsample_bytree, min_child_weight, lambda,
         alpha, gamma, split, nround, train_Rsquared_carret, test_Rsquared_carret) %>%
  mutate(r2_diff = train_Rsquared_carret - test_Rsquared_carret) %>% 
  arrange(r2_diff, desc(test_Rsquared_carret))


# just du the same SHAP stuff for SRT here 

all_model_results_SRT <-  all_model_results_SRT %>% mutate(diff_RMSE = train_RMSE - test_RMSE,
                                                              diff_R2 = train_Rsquared_carret - test_Rsquared_carret) 

all_model_results_SRT <- all_model_results_SRT %>% mutate(row_nr2 = 1:nrow(all_model_results_SRT)) 

df_loop.SRT <- all_model_results_SRT %>% filter(test_RMSE < 1.3) %>% filter(diff_RMSE < .1) %>% filter(diff_R2 < .2)  %>% filter(test_Rsquared_carret > .5)

# features <- c("PTA_full", "GMS_training", "age", "BDS_average")
features <- c("PTA_full", "GMS_training", "age", "BDS_average")
target <- "SRT"

# Load the SRT data and remove rows with missing SRT values.
data_SRT_GMS_training <- ind_diff_results %>% 
  dplyr::select(PTA_full, GMS_training, age, BDS_average, SRT) %>% 
  filter(!is.na(SRT)) %>% 
  as.data.frame()

# Use this dataset for model training.
data.model <- data_SRT_GMS_training

# clear any .Random.seed
rm(list=".Random.seed", envir=globalenv())

# limit to top 80 rows
df_top20_SRT <- df_loop.SRT[1:20, ]

n_iterations = 500

# ------ Calculate SHAP Values for Top 20 Configurations ------

# Initialize a list to store SHAP results for each configuration.
all_shap_results_SRT <- list()

# Loop over each configuration (each row in df_top20.correct).
for(i in seq_len(nrow(df_top20_SRT))) {
  
  # Extract current configuration parameters.
  curr_row <- df_top20_SRT[i, ]
  
  current_model_params <- list(
    eta              = curr_row[["eta"]], 
    max_depth        = curr_row[["max_depth"]],
    subsample        = curr_row[["subsample"]],
    colsample_bytree = curr_row[["colsample_bytree"]],
    min_child_weight = curr_row[["min_child_weight"]],
    lambda           = curr_row[["lambda"]],
    alpha            = curr_row[["alpha"]],
    gamma            = curr_row[["gamma"]]
  )
  
  # Extract model settings from the current configuration.
  bestmodel_split_current   <- curr_row[["split"]]
  bestmodel_nrounds_current <- curr_row[["nround"]]
  
  # Fit the xgboost model on the full dataset using the current configuration.
  shap_model <- xgboost::xgboost(
    data = as.matrix(data.model[, -5]),
    label = data.model[, 5],
    params = current_model_params,
    nround = bestmodel_nrounds_current,
    verbose = 0
  )
  
  # Calculate SHAP values using kernelshap and shapviz on the full dataset.
  shap_s <- kernelshap::kernelshap(shap_model, 
                                   X = as.matrix(data.model[, -5]), 
                                   bg_X = as.matrix(data.model[, -5]))
  shap_sv <- shapviz::shapviz(shap_s)
  
  # Extract SHAP values and rename columns to reflect corresponding predictors.
  tmp_SHAP <- shap_sv$S %>% 
    as.data.frame() %>% 
    dplyr::rename(
      SHAP_BDS = BDS_average,
      SHAP_GMS = GMS_training,
      SHAP_AGE = age,
      SHAP_PTA = PTA_full
    )
  
  # Calculate the average absolute SHAP values for each predictor across all observations.
  avg_SHAP <- colMeans(abs(tmp_SHAP[, c("SHAP_PTA", "SHAP_GMS", "SHAP_AGE", "SHAP_BDS")]), na.rm = TRUE)
  
  # Compute relative SHAP importance (proportion of the total absolute SHAP).
  total_abs <- sum(avg_SHAP)
  rel_SHAP <- avg_SHAP / total_abs
  
  # Create a data frame containing the calculated SHAP values.
  shap_df <- data.frame(
    SHAP_PTA     = avg_SHAP["SHAP_PTA"],
    SHAP_GMS     = avg_SHAP["SHAP_GMS"],
    SHAP_AGE     = avg_SHAP["SHAP_AGE"],
    SHAP_BDS     = avg_SHAP["SHAP_BDS"]
  )
  
  # Store the SHAP values using an identifier based on the row index.
  key_name <- paste0("Row_", i)
  all_shap_results_SRT[[key_name]] <- shap_df
}

# Combine SHAP results from all configurations into one data frame.
shap_results_combined_SRT <- dplyr::bind_rows(all_shap_results_SRT, .id = "Row_ID")

# ------ Merge SHAP Values with Top 20 Configurations ------

# Add a Row_ID column to df_top20.correct to match the identifiers used above.
df_top20_SRT <- df_top20_SRT %>% 
  mutate(Row_ID = paste0("Row_", seq_len(nrow(df_top20_SRT))))

# Merge the original top 20 configurations with the SHAP values.
df_top20_with_SHAP_SRT <- merge(df_top20_SRT, shap_results_combined_SRT, by = "Row_ID", all.x = TRUE)

# and then the same for the SRT plots....
