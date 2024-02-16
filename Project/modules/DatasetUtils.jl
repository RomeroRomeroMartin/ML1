module DatasetUtils

##########################################################

# This module provides functionalities for downloading 
# and loading datasets from various online sources.

###########################################################

using Downloads;
using DelimitedFiles;
    

function load_data(filename::String, url::String)
    if !isfile(filename)
        Downloads.download(url, filename)
    end
    data = readdlm(filename,',');
end


export load_data;  # Exports all functions
end  # Module end