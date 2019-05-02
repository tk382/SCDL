#' Wrapper function for the autoencoder prediction + filtering step
#'
#' @inheritParams preprocessDat
#' @param use.pretrain Use a pretrained model or not
#' @param pretrained.weights.file If a pretrained model is used, provide the file storing the autoencoder model weights. It should have an extension of ".hdf5" and is the saved weights from the Python package \code{sctransfer}
#' @param save.ori Whether save the original.file.name to a new file
#' @param clearup.python.session Whether to clear up everything in the Python session after computation or not. This clears up everything in Python, so you need to start a new R session to run \code{saverx} function again.
#' @param ... more arguments passed to \code{autoFilterCV}
#' @param is.large.data If the data is very large, it may take too much RAM and setting this parameter to True can reduce RAM by writing intermediate Python ouput files to disk instead of directly passing it to R. However, setting this to True can increase the computation time
#' @param batch_size batch size of the autoencoder. Default is NULL, where the batch size is automatically determined by \code{max(number of cells / 50, 32)}
#' 
#' @return RDS file saved for the autoencoder prediction + filtering result
#' @export
computePrediction <- function(text.file.name,
							  model.nodes.ID = NULL,
                is.large.data = F,
                clearup.python.session = T,
                batch_size = NULL,
							  ...) {
  ### import Python module ###
  sctransfer_tae <- reticulate::import("working_codes")
  
	### inpute checking  ###
	format <- strsplit(text.file.name, '[.]')[[1]]
	format <- paste(".", format[length(format)], sep = "")
	if (format != ".txt" && format != ".csv" && format != ".rds"){
	  stop("Input file must be in .txt or .csv or .rds form", call.=FALSE)
	}
	print(paste("Input file is:", text.file.name))
	data.species = "Human"

	main <- reticulate::import_main(convert = F)
	print("Python module sctransfer_tae imported ...")
	######

	preprocessDat(text.file.name,
				  model.nodes.ID = model.nodes.ID)
	

	out_dir <- strsplit(text.file.name, split = "/")[[1]]
	out_dir <- paste(out_dir[-length(out_dir)], collapse = "/")
	if (out_dir == ""){
	  out_dir <- "."
	}
	print("Data preprocessed ...")
	######

	### run autoencoder ###
  if (is.large.data){
    write_output_to_tsv <- T
  }else{
    write_output_to_tsv <- F
  }



	data <- readRDS(gsub(format, "_temp.rds", text.file.name))
  if (is.null(batch_size)){
    batch_size <- as.integer(max(ncol(data$mat) / 50, 32))
  } else{
    batch_size <- as.integer(batch_size)
  }
  used.time <- system.time(result <- autoFilterCV(data$mat,
												 sctransfer_tae,
												 main,
												 out_dir = out_dir,
                         batch_size = batch_size,
                         write_output_to_tsv = write_output_to_tsv))
    rm(data)
    gc()

		print(paste("Autoencoder total computing time is:", used.time[3], "seconds"))

		print(paste("Number of predictive genes is", sum(result$err.const > result$err.autoencoder)))

  if (clearup.python.session) {
    reticulate::py_run_string("
import sys
sys.modules[__name__].__dict__.clear()")
    print("Python module cleared up.")
  }

  

  temp.name <- gsub(format, "prediction.rds", text.file.name)
	saveRDS(result, file = temp.name)
  print(paste("Predicted + filtered results saved as:", temp.name))
  try(file.remove(paste0(out_dir, "/SAVERX_temp.mtx")))
  try(file.remove(paste0(out_dir, "/SAVERX_temp_test.mtx")))
  if (is.large.data) {
    try(file.remove(paste0(out_dir, "/SAVERX_temp_mean_norm.tsv")))
    try(file.remove(paste0(out_dir, "/SAVERX_temp_pred_mean_norm.tsv")))
    try(file.remove(paste0(out_dir, "/SAVERX_temp_dispersion.tsv")))
  }
  return(result)
}








