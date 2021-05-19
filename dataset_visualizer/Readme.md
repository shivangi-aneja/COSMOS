## Web Tool to visualize COSMOS dataset using the provided JSON file
This files in this directory are used only to visualize the dataset and are not for training/evaluation of models. 

**(Requirement)** Please install some SQL-based database (MySQL/SqlLite) before running the scripts provided in this repo. No other libraries need to be installed if you have already installed the libraries listed in [requirements.txt](../requirements.txt) from the home directory   

This sub-repo is a simple web application based on Python-Flask to visualize the dataset provided by the json files.   
1. **Setting up the database:** Open the file `db_cursor.py` and update the database details where you want to create and store the data. This connection would be used throughout the application for accessing data.

2. **Inserting in the database:**  Simply run the file `create_and_insert_into_db.py` to create and insert data in the tables. This might take a while to execute, since we are inserting 450K records. Also don't forget to specify path to json files in the script before executing. 

3. **Visualize Data:** Copy the train/val/test images to the [static](static/) folder or creating a symlink to the images also works.  It is very inefficient and time-consuming to load the entire dataset in one go. So we used simple pagination to do the trick. This code is configured to visualize 100 entries per page, to change this parameter simply change `per_page` parameter in the visualizer files. To visualize training/validation data, use `train_val_visualizer.py` and to view test data use `test_visualizer.py`. Exact commands to run via terminal are:

   Train Data - `python train_val_visualizer.py -m train`   
   Val Data - `python train_val_visualizer.py -m val`  
   Test Data - `python test_visualizer.py`
   

