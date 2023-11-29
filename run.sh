# a helper function to download the data
function download_data {
    [ -d data ] ||
    wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    # if data folder exists, do nothing
    mkdir data
    mv TinyStories_all_data.tar.gz data
    cd data
    tar -xvf TinyStories_all_data.tar.gz
    rm TinyStories_all_data.tar.gz
}

function parse_stories_into_file {
    # parse the stories into a single file
    # if the file exists, do nothing
    [ -f stories.txt ] ||
    python3 parse_stories.py
}

download_data
parse_stories_into_file
