#!/usr/bin/env Rscript

# Compute C-Measures for each data set and save results as CSV file.


library('tidyverse')
library("ECoL")

################################################################################
# Command line arguments
# 1: data path
# 2: output path
# 3: dataname
################################################################################

computeCMeasure = function(file_list, path_data, path_output, n) {
  is_first = TRUE
  j = 1
  for (file in file_list) {
    start = Sys.time()
    
    path_data_ = sprintf('%s%s',path_data, file)
    print(sprintf('Read from: %s', path_data_))

    data = read.csv(path_data_)
    data$y = as.factor(data$y)
    n_class = length(levels(data$y))
    if (n_class != 2) {
      print(sprintf('Expecting 2 classes, Got %d. Next.', n_class))
      next
    }
    tryCatch({
      c_measure = complexity(y~., data)
      attr = c(file)
      names(attr) = c('Data')
      c_measure <- c(attr, c_measure)

      if (is_first) {
        df = data.frame(t(c_measure))
        is_first = FALSE
      } else {
        df = rbind(df, data.frame(t(c_measure)))
      }
    }, warning = function(warn) {
      print(paste('Complexity Warning:', warn))
    }, error = function(err) {
      print(paste('Complexity Error:', err))
    })
    
    time_elapsed = as.numeric(Sys.time() - start, units="secs")
    print(sprintf('[%d] Time: %3.0fs File: %s', j, time_elapsed, file))
    j = j + 1
  }
  
  print(path_output)
  write.csv(df, path_output, row.names=FALSE)
}


args = commandArgs(trailingOnly=TRUE)

if (length(args) == 2) {
    path_data = args[1]
    path_output = args[2]
} else {
  error_msg = sprintf('2 is required (1:clean data path. 2: output path.). Got %d', length(args))
  stop(error_msg, call.=FALSE)
}

print(normalizePath(path_data))
# Matching pattern: <dataname><something in middle>.csv
reg_exp = '[a-zA-Z0-9_.]+(.csv)$'
file_list = list.files(path=path_data, pattern=reg_exp)
print(sprintf('# of datasets: %d', length(file_list)))
computeCMeasure(file_list, path_data, path_output, n)
