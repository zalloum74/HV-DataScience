##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Load libraries
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)

# Understand data 
str(edx)
summary(edx)

# check for NA value 
anyNA(edx)


# Number of movies and users
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  

# Number of unique movies and users
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# Number of ratings by movie 
edx %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue",bins = 30 , binwidth = 0.2)+
  scale_x_log10()+
  ggtitle(" Number of Rating per Movie")+
  theme_gray()


# Number of ratings by user 
edx %>% count(userId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue" , bins = 30, binwidth = 0.2)+
  ggtitle(" Number of Rating Per User")+
  scale_x_log10()+
  theme_gray()


# Rating by movie genres 
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
  geom_bar(aes(fill =genres),stat = "identity")+ 
  labs(title = " Number of Rating for Each Genre")+
  theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))+
  theme_light()


# Top 10 genre 
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Data partition
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]


# Calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}


# Model 1
Mu_1 <- mean(train_set$rating)
Mu_1

naive_rmse <- RMSE(test_set$rating,Mu_1)
naive_rmse

# Get the RMSE and compare results.
rmse_results <- data_frame(method = "The average", RMSE = naive_rmse)
rmse_results%>% knitr::kable()


# Model 2
Mu_2 <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - Mu_2))


# Visulize results
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))


# Prediction enhancement:
predicted_ratings <- Mu_2 + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_2_rmse))
rmse_results %>% knitr::kable()



# Model 3
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")


# Calculate the mean
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - Mu_2 - b_i))

# Calculate RMSE again to see the enhancement 

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = Mu_2 + b_i + b_u) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results%>% knitr::kable()


# Apply RMSE for validation
valid_pred_rating <- validation %>%
  left_join(movie_avgs , by = "movieId" ) %>% 
  left_join(user_avgs , by = "userId") %>%
  mutate(pred = Mu_2 + b_i + b_u ) %>%
  pull(pred)
model_3_valid <- RMSE(validation$rating, valid_pred_rating)
rmse_results <-  bind_rows( rmse_results, data_frame(Method = "Validation Results" , RMSE = model_3_valid))
rmse_results%>% knitr::kable()