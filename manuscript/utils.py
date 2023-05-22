import json

def get_directories(directories_file: str):
    """
    Reads a text file with directories for input and output of the analysis.
    :param directories_file: .txt file with {md_dir, md_analysis_dir, desc_dir, desc_dir_dend, pcm_dir} specified in
    the format: md_dir = 'C:\path\to\directory'
    :return: paths to directories
    """
    try:
        with open(directories_file, 'r') as f:
            directories_dict = json.load(f)
    except FileNotFoundError:
        print('Define a .json file with the five directories needed for the analysis:')
        print('md_dir, md_analysis_dir, desc_dir, desc_dir_dend and pcm_dir')
        print('The format of the file must be a dictionary.')

    try:
        return directories_dict['md_dir'], directories_dict['md_analysis_dir'], directories_dict['desc_dir'], \
           directories_dict['desc_dir_dend'], directories_dict['pcm_dir'], directories_dict['papyrus_dir']
    except KeyError:
        print('The following directories need to be defined: md_dir, md_analysis_dir, desc_dir, desc_dir_dend and pcm_dir')
