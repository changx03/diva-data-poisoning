#!/usr/bin/env Rscript

# Compute C-Measures for each data set and save results as CSV file.

library('tidyverse')
library("ECoL")

################################################################################
# Parameters
n = 100  # Save every 100 files
################################################################################

computeCMeasure = function(file_list, path_data, path_output, n) {
  stop = length(file_list)/n - 1
  j = 1
  for (i in 0:stop) {  
    start = 1+i*n
    end = (i+1)*n
    files = file_list[start:end]
    print(sprintf('From %d to %d', start, end))
    
    is_first = TRUE
    for (file in files) {
      print(file)
      start = Sys.time()
      
      data = read.csv(paste(path_data, file, sep=''))
      data$y = as.factor(data$y)
      
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
    folder_name = args[1]
    folder_poison = args[2]
    output_name = args[3]
} else {
  stop('At least 3 argument must be supplied (1:clean data path; 2: poison data path; 3: output filename).n', call.=FALSE)
}

print('Computing C-Measures for clean data')
path_data = folder_name
print(normalizePath(path_data))
file_list = list.files(path=path_data, pattern='*.csv')
print(sprintf('# of datasets: %d', length(file_list)))
path_output = sprintf('../results/%s', output_name)
computeCMeasure(file_list, path_data, path_output, n)

print('Computing C-Measures for poisoned data')
path_data = sprintf('%s/%s/', folder_name, folder_poison)
print(path_data)
file_list = list.files(path=path_data, pattern='*.csv')
print(sprintf('# of datasets: %d', length(file_list)))
path_output = sprintf('../results/%s', output_name)
computeCMeasure(file_list, path_data, path_output, n)
