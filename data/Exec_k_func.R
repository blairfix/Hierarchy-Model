library(data.table)
library(readr)
library(magrittr)
library(here)


dir = here()
workspace = gsub("Hierarchy Model/data",  "Emperical Data/Execucomp/Results", dir)
setwd(workspace)


# Exec capitalist income vs firm size regression 
################################################

Exec = fread("Exec_power.csv")  

#log regression of labor frac vs. power
y = Exec$K.frac
x = log( Exec$Emp)
r = lm( y ~ x + 0)
const = as.numeric(coef(r)) 


# mean capitalist income
########################
workspace = gsub("Hierarchy Model/data",  "Emperical Data/IPUM", dir)
setwd(workspace)

stats = fread("k_stats.csv")  
stats = stats[Year < 2015]
mean.year = stats[, mean(Mean), by = Year]


div_min = min(mean.year$V1)/1000
div_max = max(mean.year$V1)/1000

output = c(const, div_min, div_max) %>% data.table()



setwd(dir)
write_tsv(output, "k_parameters.txt", col_names = F)




