library(data.table)
library(readr)
library(here)
library(magrittr)


dir = here()


# Format Compustat data
####################################
workspace = gsub("Hierarchy Model/data",  "Emperical Data/Execucomp/Results", dir)
setwd(workspace)

d = fread("Exec_model_input.csv")

pay.norm = d[ , mean.pay/mean(mean.pay), by = Year] # normalize mean pay to have average of one in each year

out = data.table(d$Employment, d$ceo_ratio, pay.norm$V1) 

setwd(dir)
write_tsv(out, "compustat.txt", col_names = F)




# Format Case Study data
###########################
workspace = gsub("Hierarchy Model/data",  "Emperical Data/Case Studies", dir)
setwd(workspace)

d = fread("Case Results.csv")

# gini index by hierarchical level
g = d$Gini %>% na.omit()
g = g[g != 0] %>% data.table()


s = data.table(d$Level, d$span) %>% na.omit()


setwd(dir)
write_tsv(g, "g_empirical.txt", col_names = F)
write_tsv(s, "s_empirical.txt", col_names = F)

# Power law percentile cuttoff
###############################
workspace = gsub("Hierarchy Model/data",  "Emperical Data/IRS", dir)
setwd(workspace)

p = fread("power_law_percentile.csv")

setwd(dir)
write_tsv(p, "power_law_percentile.tsv", col_names = F)



