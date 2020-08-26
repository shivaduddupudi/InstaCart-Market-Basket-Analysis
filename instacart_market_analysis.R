# Instacart Market Basket Analysis

# Load Required Packages
library(data.table)
library(h2o)

# Initialization parameters for h2o
h2o.init(nthreads = 15, max_mem_size = "16g")

# Load and store datasets
ailes <- fread("aisles.csv", key = "aisle_id")
department <- fread("departments.csv", key = "department_id")
product <- fread("products.csv", key = c("product_id","aisle_id", "department_id"))
order_prod_prior <- fread("order_products__prior.csv")
order_prod_train <- fread("order_products__train.csv")
orders <- fread("orders.csv")

# Join aisle and department information onto 'product' table
product <- merge(product, ailes, by = "aisle_id", all.x = TRUE, sort = FALSE)#left outer join
product <- merge(product, department, by = "department_id", all.x = TRUE, sort = FALSE)

# Join 'prod' and 'order' tables onto prior orders table
order_prod_prior <- merge(order_prod_prior, product, by = "product_id", all.x = TRUE, sort = FALSE)
order_prod_prior <- merge(order_prod_prior, orders, by = "order_id", all.x = TRUE, sort = FALSE)

# Shows how many orders ago a particular product was ordered
order_prod_prior[,":="(orders_ago = max(order_number) - order_number + 1), by = user_id]

# Create a list of all prior products purchased for each user
user_prod_list <- order_prod_prior[ ,.(last_order_number = max(order_number), purch_count = .N), keyby = .(user_id, product_id)]

# Feature engineering
user_summ <- order_prod_prior[,.(user_total_products_ordered_hist = .N, # Total products ordered per user
                                 uniq_prod = uniqueN(product_name), # Unique products ordered per user
                                 uniq_aisle = uniqueN(aisle), # Unique aisles ordered from per user
                                 uniq_dept = uniqueN(department), # Unique departments ordered from per user
                                 prior_orders = max(order_number)), # Total number of orders per user
                              by = user_id]

user_prior_prod_cnt <- order_prod_prior[,.(prior_prod_cnt = .N, # Count of prior products ordered per user
                                           last_purchased_orders_ago = min(orders_ago), # Number of orders ago a product was last puchases
                                           first_purchased_orders_ago = max(orders_ago)), # Number of orders ago a product was first purchases
                                        by=.(user_id, product_id)]

# Join tables to create training set
opt_user <- merge(order_prod_train[reordered == 1, .(order_id, product_id)], orders[,.(order_id, user_id)], by = "order_id", all.x = TRUE, sort = FALSE)
dt_expanded  <- merge(user_prod_list[user_id %in% opt_user[["user_id"]],.(user_id, product_id)], opt_user, by = c("user_id", "product_id"), all.x = TRUE, sort = FALSE)
dt_expanded[, curr_prod_purchased:=ifelse(!is.na(order_id), 1, 0)]

train <- merge(dt_expanded, user_summ, by="user_id", all.x = TRUE, sort = FALSE)
train <- merge(train, user_prior_prod_cnt, by = c("user_id", "product_id"), all.x = TRUE, sort = FALSE)
varnames <- setdiff(colnames(train), c("user_id","order_id","curr_prod_purchased"))

# Join tables to create test set
test_orders <- orders[eval_set == "test"]
optest_user <- merge(test_orders[,.(order_id)], orders[,.(order_id, user_id)], by = "order_id", all.x = TRUE, sort = FALSE)
dt_expanded  <- merge(user_prod_list[user_id %in% optest_user[["user_id"]],.(user_id, product_id)], optest_user, by = c("user_id"), all.x = TRUE, sort = FALSE)
dt_expanded[,curr_prod_purchased:=sample(c(0,1), nrow(dt_expanded), replace = TRUE)] 
test <- merge(dt_expanded, user_summ, by="user_id", all.x = TRUE, sort = FALSE)
test <- merge(test, user_prior_prod_cnt, by=c("user_id", "product_id"), all.x = TRUE, sort = FALSE)

# Sample 10000 entries from the training set for a validation set
set.seed(5082)
val_users <- sample(unique(train$user_id), size = 10000, replace = FALSE)

# Convert response variable data type to factor
train[,curr_prod_purchased:=as.factor(curr_prod_purchased)]
test[,curr_prod_purchased:=as.factor(curr_prod_purchased)]

# Add data to h2o module
train.hex <- as.h2o(train[!user_id %in% val_users,c("curr_prod_purchased", varnames), with = FALSE], destination_frame = "train.hex")
val.hex <- as.h2o(train[user_id %in% val_users,c("curr_prod_purchased", varnames), with = FALSE], destination_frame = "val.hex")

# Delete old, unused tables to free up some memory
rm(train, order_prod_prior, order_prod_train, orders, product, department, ailes, user_prod_list, user_summ);gc()

# Train random forest model
randf <- h2o.randomForest(x = varnames, # Feature space
                          y = "curr_prod_purchased", # Response variable
                          training_frame = train.hex, # Training data
                          validation_frame = val.hex, # Validation data
                          model_id = "T2_rf_model", # Model object name
                          nfolds = 3, # Number of cross-validation folds
                          ntrees = 50, # Number of trees to build
                          max_depth = 6, # Max depth of each tree
                          stopping_rounds = 3, # Number of rounds to stop after if evaluation metric does not improve
                          seed = 5082, # Random seed
                          mtries = -1, # Number of features to sample (-1 means default sqrt(p) for classification or p/3 for regression)
                          stopping_metric = "logloss" # Evaluation/stopping metric
)

# Make predictions of test set
test.hex <- as.h2o(test[, c("curr_prod_purchased", varnames), with = FALSE], destination_frame = "test.hex")

sPreds <- as.data.table(h2o.predict(randf, test.hex))
sPreds <- data.table(order_id = test$order_id, product_id = test$product_id, testPreds = sPreds$p1)
testPreds <- sPreds[,.(products = paste0(product_id[testPreds > 0.21], collapse = " ")), by = order_id]
set(testPreds, which(testPreds[["products"]] == ""), "products", "None")

# Print results and write to CSV for submission
testPreds
fwrite(testPreds, "submission.csv")