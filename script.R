# Title: Seoul Bike Sharing Demand Prediction
# Author: Ivan Blanquez
# Date: February 24, 2025

# Detect operating system and set target directory
cat("Detecting operating system...\n")
os <- Sys.info()["sysname"]
cat("Operating system:", os, "\n")
if (os == "Windows") {
  target_dir <- "C:\\RSTUDIO"
} else {
  target_dir <- "~/RSTUDIO"
  cat("Using", target_dir, "since C:\\RSTUDIO is Windows-specific.\n")
}

# Create directory if it doesn’t exist
cat("Ensuring directory", target_dir, "exists...\n")
if (!dir.exists(target_dir)) {
  dir.create(target_dir, showWarnings = TRUE)
  if (!dir.exists(target_dir)) {
    stop("Failed to create", target_dir, ". Run with admin rights or choose a writable directory.")
  }
}
setwd(target_dir)
cat("Working directory set to:", getwd(), "\n")

# Install and load required packages
cat("Checking and installing required packages...\n")
required_packages <- c("tidyverse", "lubridate", "randomForest", "xgboost", "caret", "Metrics", "ggplot2", "GGally")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if (length(new_packages)) {
  cat("Installing:", paste(new_packages, collapse = ", "), "\n")
  tryCatch(
    install.packages(new_packages, dependencies = TRUE),
    error = function(e) {
      cat("Default install failed:", e$message, "\nTrying user library...\n")
      user_lib <- file.path(Sys.getenv("HOME"), "R_libs")
      if (!dir.exists(user_lib)) dir.create(user_lib)
      install.packages(new_packages, dependencies = TRUE, lib = user_lib)
      .libPaths(c(user_lib, .libPaths()))
    }
  )
}

cat("Loading libraries...\n")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    stop("Failed to load package:", pkg, ". Install it manually with install.packages('", pkg, "')")
  }
}
cat("All libraries loaded successfully.\n")

# Set seed for reproducibility
set.seed(1234)

# 1. Data Acquisition
cat("Downloading dataset...\n")
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
download.file(url, destfile = "SeoulBikeData.csv")
data <- read_csv("SeoulBikeData.csv", col_types = cols(Date = col_date(format = "%d/%m/%Y")))
cat("Dataset loaded. Rows:", nrow(data), "\n")
cat("Raw column names:", paste(colnames(data), collapse = ", "), "\n")

# 2. Data Cleaning and Feature Engineering
cat("Cleaning data...\n")
temp_col <- grep("Temperature", colnames(data), value = TRUE)
if (length(temp_col) == 0) {
  cat("WARNING: 'Temperature' not found with grep. Using column 4.\n")
  temp_col <- colnames(data)[4]
}
dewpoint_col <- grep("Dew point temperature", colnames(data), value = TRUE)
if (length(dewpoint_col) == 0) {
  cat("WARNING: 'Dew point temperature' not found with grep. Using column 8.\n")
  dewpoint_col <- colnames(data)[8]
}
cat("Using Temperature column:", temp_col, "\n")
cat("Using Dew point temperature column:", dewpoint_col, "\n")

data_clean <- data %>%
  rename(BikeCount = `Rented Bike Count`,
         Temp = !!temp_col,
         DewPoint = !!dewpoint_col,
         Rain = "Rainfall(mm)",
         Humid = "Humidity(%)",
         WindSpeed = "Wind speed (m/s)",
         Visibility = "Visibility (10m)",
         SolarRad = "Solar Radiation (MJ/m2)",
         Snow = "Snowfall (cm)") %>%
  mutate(DayOfWeek = as.numeric(wday(Date, label = TRUE)),
         HourSin = sin(2 * pi * Hour / 24),
         HourCos = cos(2 * pi * Hour / 24),
         BikeCount = pmin(BikeCount, quantile(BikeCount, 0.99))) %>%
  select(-Date) %>%
  mutate_at(vars(Seasons, Holiday, `Functioning Day`), as.factor)
cat("Cleaned column names:", paste(colnames(data_clean), collapse = ", "), "\n")

data_encoded <- dummyVars("~ Seasons + Holiday + `Functioning Day`", data = data_clean) %>%
  predict(data_clean) %>%
  as.data.frame()
colnames(data_encoded) <- make.names(colnames(data_encoded), unique = TRUE)
data_encoded <- data_encoded %>%
  bind_cols(data_clean %>% select(-Seasons, -Holiday, -`Functioning Day`))
cat("Encoded column names:", paste(colnames(data_encoded), collapse = ", "), "\n")
cat("Data cleaned. Columns:", ncol(data_encoded), "\n")

# 3. Exploratory Data Analysis
cat("Generating EDA plots...\n")
p1 <- ggplot(data_clean, aes(x = Hour, y = BikeCount)) +
  geom_boxplot() +
  labs(title = "Hourly Bike Demand Distribution", x = "Hour of Day", y = "Bike Count") +
  theme_minimal()
ggsave("figure1_hourly_demand.png", p1, width = 8, height = 6)

p2 <- ggpairs(data_clean %>% select(BikeCount, Temp, Rain, Humid),
              title = "Scatterplot Matrix of Key Variables") +
  theme_minimal()
ggsave("figure2_scatterplot_matrix.png", p2, width = 10, height = 10)
cat("EDA plots saved to", getwd(), "\n")

# 4. Train-Test Split
cat("Splitting data...\n")
trainIndex <- createDataPartition(data_encoded$BikeCount, p = 0.8, list = FALSE)
train <- data_encoded[trainIndex, ]
test <- data_encoded[-trainIndex, ]
X_train <- train %>% select(-BikeCount) %>% as.matrix()
y_train <- train$BikeCount
X_test <- test %>% select(-BikeCount) %>% as.matrix()
y_test <- test$BikeCount
cat("Train rows:", nrow(train), "Test rows:", nrow(test), "\n")
cat("X_train class:", class(X_train), "dimensions:", dim(X_train), "\n")

# 5. Model 1: Random Forest
cat("Training Random Forest...\n")
rf_model <- randomForest(BikeCount ~ ., data = train, ntree = 500, maxdepth = 10)
rf_pred <- predict(rf_model, test)
rf_rmse <- rmse(y_test, rf_pred)
rf_mae <- mae(y_test, rf_pred)
cat("Random Forest trained. Predictions length:", length(rf_pred), "\n")

# 6. Model 2: XGBoost
cat("Training XGBoost...\n")
xgb_data <- xgb.DMatrix(data = X_train, label = y_train)
xgb_model <- xgb.train(params = list(objective = "reg:squarederror", max_depth = 6, eta = 0.1),
                       data = xgb_data, nrounds = 200)
xgb_pred <- predict(xgb_model, X_test)
xgb_rmse <- rmse(y_test, xgb_pred)
xgb_mae <- mae(y_test, xgb_pred)
cat("XGBoost trained. Predictions length:", length(xgb_pred), "\n")

# 7. Results Visualization
cat("Generating visualizations...\n")
results <- data.frame(Actual = y_test, RF_Pred = rf_pred, XGB_Pred = xgb_pred)
p3 <- ggplot(results, aes(x = Actual)) +
  geom_point(aes(y = RF_Pred, color = "Random Forest"), alpha = 0.5) +
  geom_point(aes(y = XGB_Pred, color = "XGBoost"), alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0) +
  labs(title = "Predicted vs. Actual Bike Counts", x = "Actual", y = "Predicted") +
  theme_minimal()
ggsave("figure3_pred_vs_actual.png", p3, width = 8, height = 6)

importance <- xgb.importance(model = xgb_model)
p4 <- ggplot(importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance (XGBoost)", x = "Feature", y = "Gain") +
  theme_minimal()
ggsave("figure4_feature_importance.png", p4, width = 8, height = 6)
cat("Visualizations saved to", getwd(), "\n")

# 8. Print Results
cat("Final Results:\n")
cat("Random Forest - RMSE:", rf_rmse, "MAE:", rf_mae, "\n")
cat("XGBoost - RMSE:", xgb_rmse, "MAE:", xgb_mae, "\n")
cat("Script execution completed. Outputs saved to:", getwd(), "\n")

# Verify outputs
cat("Verifying output files...\n")
expected_files <- c("SeoulBikeData.csv", "figure1_hourly_demand.png", "figure2_scatterplot_matrix.png",
                    "figure3_pred_vs_actual.png", "figure4_feature_importance.png")
for (file in expected_files) {
  if (file.exists(file)) {
    cat(file, "found.\n")
  } else {
    cat("WARNING:", file, "not found.\n")
  }
}