##########################################################
# Create edx and final_holdout_test sets 
##########################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(stringr)
library(dplyr)
library(recosystem) 
library(tidyr)



# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

file.info("ml-10M100K.zip")$size
​
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
​
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# EXTRAER TODO EL ZIP
unzip(dl, exdir = ".")

ratings_file <- "ml-10M100K/ratings.dat"
movies_file  <- "ml-10M100K/movies.dat"
​
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
​
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
​
movielens <- left_join(ratings, movies, by = "movieId")
​
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
​
# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
​
# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
​
rm(dl, ratings, movies, test_index, temp, movielens, removed)

dim(edx)
nrow(final_holdout_test)


sum(edx$rating == 0)
sum(edx$rating == 3)

length(unique(edx$movieId))

length(unique(edx$userId))


sum(str_detect(edx$genres, "Drama"))

sum(str_detect(edx$genres, "Comedy"))

sum(str_detect(edx$genres, "Thriller"))

sum(str_detect(edx$genres, "Romance"))

edx %>% 
  group_by(title) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  head()

edx %>% 
  count(rating) %>% 
  arrange(desc(n))



# Define RMSE
RMSE <- function(true, pred){
  sqrt(mean((true - pred)^2))
}

# FIRST  METHOD: Regularized Baseline Predictor




# Model 1: Global Average Baseline

mu_hat <- mean(edx$rating)

pred_mean <- rep(mu_hat, nrow(final_holdout_test))

rmse_mean <- RMSE(final_holdout_test$rating, pred_mean)
rmse_mean

# Model 2: Movie Effect Model

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

pred_movie <- final_holdout_test %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  mutate(pred = mu_hat + b_i) %>% 
  pull(pred)

rmse_movie <- RMSE(final_holdout_test$rating, pred_movie)
rmse_movie

# Model 3: Movie and User Effects

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

user_avgs <- edx %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu_hat - b_i))

pred_movie_user <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

rmse_movie_user <- RMSE(final_holdout_test$rating, pred_movie_user)
rmse_movie_user

# Model 4: Regularized Movie + User effects


set.seed(1, sample.kind = "Rounding")

# Internal train/validation split

val_index <- createDataPartition(edx$rating, p = 0.1, list = FALSE)
train <- edx[-val_index, ]
val   <- edx[val_index, ]

# Lambda Tunning

lambdas <- seq(1, 10, 0.5)

rmse_lambda <- sapply(lambdas, function(l) {
  
  mu_t <- mean(train$rating, na.rm = TRUE)
  
  b_i_t <- train %>%
    group_by(movieId) %>%
    summarize(
      b_i = sum(rating - mu_t, na.rm = TRUE) / (n() + l),
      .groups = "drop"
    )
  
  b_u_t <- train %>%
    left_join(b_i_t, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      b_u = sum(rating - mu_t - b_i, na.rm = TRUE) / (n() + l),
      .groups = "drop"
    )
  
  pred_val_tbl <- val %>%
    left_join(b_i_t, by = "movieId") %>%
    left_join(b_u_t, by = "userId") %>%
    replace_na(list(b_i = 0, b_u = 0)) %>%
    mutate(pred = mu_t + b_i + b_u)
  
  RMSE(pred_val_tbl$rating, pred_val_tbl$pred)
})

rmse_lambda
best_lambda <- lambdas[which.min(rmse_lambda)]
best_lambda

# Final model training with best lambda

# 1. Choose a reasonable lambda manually
lambda <- best_lambda

# 2. Global mean
mu <- mean(edx$rating, na.rm = TRUE)

# 3. Regularized movie effect b_i
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(
    b_i = sum(rating - mu, na.rm = TRUE) / (n() + lambda),
    .groups = "drop"
  )

# 4. Regularized user effect b_u
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    b_u = sum(rating - mu - b_i, na.rm = TRUE) / (n() + lambda),
    .groups = "drop"
  )

# 5. Build predictions on your existing final_holdout_test
pred_tbl <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  replace_na(list(b_i = 0, b_u = 0)) %>%
  mutate(pred = mu + b_i + b_u)


# 6. RMSE of the regularized baseline predictor
rmse_reg <- RMSE(pred_tbl$rating, pred_tbl$pred)
rmse_reg

# SECOND METHOD: Matrix factorization


# Preparing the data
edx <- edx %>%
  mutate(
    userId  = as.integer(userId),
    movieId = as.integer(movieId),
    rating  = as.numeric(rating)
  )

final_holdout_test <- final_holdout_test %>%
  mutate(
    userId  = as.integer(userId),
    movieId = as.integer(movieId),
    rating  = as.numeric(rating)
  )

# Create objets for recosystem package
train_data <- data_memory(
  user_index = edx$userId,
  item_index = edx$movieId,
  rating     = edx$rating
)


test_data <- data_memory(
  user_index = final_holdout_test$userId,
  item_index = final_holdout_test$movieId
)


#Initialize model and tuning of hyperparameters

set.seed(1)  

r <- Reco()  # Initialize factorization model

opts_tune <- r$tune(
  train_data,
  opts = list(
    dim       = c(10, 20, 50),  # probar distintas dimensiones latentes
    lrate     = c(0.05, 0.1),   # distintos learning rates
    costp_l2  = c(0.01, 0.1),   # regularización para usuarios
    costq_l2  = c(0.01, 0.1),   # regularización para ítems
    nthread   = 4,              # ajusta según tu CPU
    niter     = 20,             # iteraciones de tuning
    verbose   = TRUE
  )
)

opts_tune$min  # best group of hyperparameters

best_opts <- opts_tune$min

r$train(
  train_data,
  opts = c(
    best_opts,
    nthread = 4,   
    niter   = 50,  
    verbose = TRUE
  )
)

# Predictions on test data
pred_ratings <- r$predict(test_data)

# RMSE
rmse_mf <- RMSE(final_holdout_test$rating, pred_ratings)
rmse_mf





