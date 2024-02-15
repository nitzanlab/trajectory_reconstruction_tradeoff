# Converting rds data file from https://zenodo.org/record/1443566\#.YEExrpMzbDI to csv files

dataname <-'fibroblasts'
filename <-'fibroblast-reprogramming_treutlein'

fname <- # <path_to_file>
outdir <- # <path_to_output_dir>

readRDS(fname) -> data
write.csv(data$cell_info , file.path(outdir, paste(dataname,'_cell_info.csv', sep='')))
write.csv(data$counts, file.path(outdir, paste('counts_', dataname,'.csv', sep='')))
write.csv(data$milestone_network, file.path(outdir, paste(dataname,'_milestone_network.csv', sep='')))

