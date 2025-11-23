# lab5_bn.R
# CS367 - Lab 5: Bayesian Networks and Naive Bayes in R

# Required packages:
# install.packages("bnlearn")
# install.packages("gRain")

library(bnlearn)
library(gRain)

# 1. Load and preprocess data

data_file <- "2020_bn_nb_data.txt"   # put this file in same folder
df <- read.table(data_file, header = TRUE, stringsAsFactors = TRUE)

# Assume columns: PH100, EC100, IT101, MA101, Internship
str(df)

# Ensure all grade columns are factors
grade_cols <- c("PH100", "EC100", "IT101", "MA101")
df[grade_cols] <- lapply(df[grade_cols], factor)

df$Internship <- factor(df$Internship)

# 2. Structure learning

# Hill-Climbing with BIC
bn_hc <- hc(df, score = "bic")
print(bn_hc)
plot(bn_hc)

# Tabu search with BIC (optional)
bn_tabu <- tabu(df, score = "bic")
print(bn_tabu)
# plot(bn_tabu)

# Choose one (say hc) for further steps
bn_model <- bn_hc

# 3. Parameter learning (CPTs)

fitted_bn <- bn.fit(bn_model, data = df)
print(fitted_bn$PH100)  # Example CPT

# 4. Inference: P(PH100 | EC100=DD, IT101=CC, MA101=CD)

# Convert to gRain object
junction_bn <- as.grain(fitted_bn)

evidence <- list(
  EC100 = "DD",
  IT101 = "CC",
  MA101 = "CD"
)

query <- querygrain(setEvidence(junction_bn, evidence = evidence),
                    nodes = "PH100", type = "distribution")

print(query)

# 5. Naive Bayes classifier for Internship

set.seed(42)

# Train-test split function (70-30)
make_split <- function(df, train_ratio = 0.7) {
  n <- nrow(df)
  idx <- sample(1:n, size = floor(train_ratio * n))
  list(train = df[idx, ], test = df[-idx, ])
}

# Naive Bayes structure: Internship -> all courses
nb_structure <- naive.bayes(x = df, class = "Internship")
print(nb_structure)
# plot(nb_structure)

runs <- 20
acc_nb <- numeric(runs)

for (r in 1:runs) {
  split <- make_split(df)
  train_df <- split$train
  test_df  <- split$test
  
  nb_fit <- bn.fit(nb_structure, data = train_df)
  nb_grain <- as.grain(nb_fit)
  
  # Predict Internship for test data
  preds <- character(nrow(test_df))
  
  for (i in 1:nrow(test_df)) {
    ev <- as.list(test_df[i, grade_cols])
    ev_grain <- setEvidence(nb_grain, evidence = ev)
    dist_I <- querygrain(ev_grain, nodes = "Internship", type = "distribution")
    probs <- dist_I$Internship
    preds[i] <- names(probs)[which.max(probs)]
  }
  
  acc_nb[r] <- mean(preds == test_df$Internship)
}

cat("Naive Bayes average accuracy over 20 runs:", mean(acc_nb), "\n")


# 6. BN classifier (full dependencies)

# Here we treat Internship as a node in the learned BN (bn_model)
# and use it as a classifier instead of naive structure.

runs <- 20
acc_bn <- numeric(runs)

for (r in 1:runs) {
  split <- make_split(df)
  train_df <- split$train
  test_df  <- split$test
  
  # Refit BN on training data only
  bn_model_train <- hc(train_df, score = "bic")
  bn_fit_train <- bn.fit(bn_model_train, data = train_df)
  bn_grain <- as.grain(bn_fit_train)
  
  preds <- character(nrow(test_df))
  
  for (i in 1:nrow(test_df)) {
    ev <- as.list(test_df[i, grade_cols])
    ev_grain <- setEvidence(bn_grain, evidence = ev)
    dist_I <- querygrain(ev_grain, nodes = "Internship", type = "distribution")
    probs <- dist_I$Internship
    preds[i] <- names(probs)[which.max(probs)]
  }
  
  acc_bn[r] <- mean(preds == test_df$Internship)
}

cat("BN classifier average accuracy over 20 runs:", mean(acc_bn), "\n")
