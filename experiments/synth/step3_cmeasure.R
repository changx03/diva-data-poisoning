#!/usr/bin/env Rscript

# Compute C-Measures for each data set and save results as CSV file.


library('tidyverse')
library("ECoL")

################################################################################
# Command line arguments
# 1: data path
# 2: output path
# 3: output name

################################################################################
# Parameters
# Save every n files. Each output CSV file contains n rows / samples.
n = 50
################################################################################

computeCMeasure = function(file_list, path_data, path_output, n) {
  stop = length(file_list)/n - 1
  if (stop < 1) {
    stop = 0
  }
  j = 1
  for (i in 0:stop) {  
    start = 1+i*n
    end = (i+1)*n
    if (end > length(file_list)) {
      end = length(file_list)
    }
    files = file_list[start:end]
    print(sprintf('From %d to %d', start, end))
    
    is_first = TRUE
    for (file in files) {
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
    
    path_output_final = sprintf('%s_%d.csv', path_output, i)
    print(path_output_final)
    write.csv(df, path_output_final, row.names=FALSE)
  }
}


args = commandArgs(trailingOnly=TRUE)

if (length(args)==3) {
    path_data = args[1]
    folder_output = args[2]
    output_name = args[3]
} else {
  error_msg = sprintf('3 arguments must be supplied (1:clean data path. 2: output path. 3: output name). Got %d', length(args))
  stop(error_msg, call.=FALSE)
}

print(normalizePath(path_data))
file_list = list.files(path=path_data, pattern='*.csv')
print(sprintf('# of datasets: %d', length(file_list)))
# path_output is NOT the full name, since we will save multiple file during training.
path_output = sprintf('%s%s', folder_output, output_name)
computeCMeasure(file_list, path_data, path_output, n)
