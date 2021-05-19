""" Helper class to save data in the database"""

import os
from urllib.parse import urlsplit
from dataset_visualizer.db_cursor import mydb, my_cursor
from utils.config import DATA_DIR
from utils.common_utils import read_json_data


def create_tables():
    """
        Creates tables for storing train, val and test data
    """
    # Source Table
    print("Creating Training set data table")
    my_cursor.execute("create table TRAIN_DATA_CAPTIONS  (id int auto_increment primary key, base_url varchar(1000), article_url varchar(2000) null, img_local_path  varchar(1000) null, caption  varchar(6000) null)")
    print("Creating Validation set data table")
    my_cursor.execute("create table VAL_DATA_CAPTIONS  (id int auto_increment primary key, base_url varchar(1000), article_url varchar(2000) null, img_local_path  varchar(1000) null, caption  varchar(6000) null)")
    print("Creating Test set data table")
    my_cursor.execute("create table TEST_DATA_CAPTIONS ( id int auto_increment primary key, caption1 varchar(5000) null, caption2 varchar(5000) null, context_label  int null, base_url varchar(1000) null, article_url varchar(2000) null, img_local_path varchar(1000) null )")


def insert_test_data(data_list, table_name):
    """
        Inserts Test data in the table name specified

        Args:
            data_list (list[dict]): Data to be stored in the database
            table_name (str): name of the database table

        Returns:
            None
    """
    for data in data_list:
        sql_insert = 'INSERT INTO ' + table_name + ' (caption1, caption2, context_label, base_url, article_url, img_local_path ) VALUES (%s, %s, %s, %s, %s, %s)'
        if "article_url" in data.keys() and len(data["article_url"]) > 0:
            base_url = "{0.netloc}".format(urlsplit(data['article_url'])).replace("www.", "")
        else:
            data["article_url"] = ""
            base_url = ""
        data_tuple = (data["caption1"], data["caption2"], data["context_label"], base_url, data["article_url"], data["img_local_path"])
        my_cursor.execute(sql_insert, data_tuple)
        mydb.commit()
    print("Inserted")


def insert_train_val_data(article_list, table_name):
    """
        Inserts Train and Val data in the table name specified

        Args:
            article_list (list[dict]): Data to be stored in the database
            table_name (str): name of the database table

        Returns:
            None
    """
    for article in article_list:
        img_path = article['img_local_path']
        articles_db = []
        for data in article['articles']:
            base_url = "{0.netloc}".format(urlsplit(data['article_url'])).replace("www.", "")
            articles_db.append((data['caption'], base_url, data['article_url'], img_path))

        sql = 'INSERT INTO ' + table_name + ' (caption, base_url, article_url, img_local_path ) VALUES (%s, %s, %s, %s)'
        my_cursor.executemany(sql, articles_db)
        mydb.commit()
        print(my_cursor.rowcount, 'inserted')


def insert_into_tables():
    """
        Inserts data in the tables
    """
    print("Inserting Training Data")
    articles = read_json_data(os.path.join(DATA_DIR, "annotations", "train_data.json"))
    insert_train_val_data(articles, "TRAIN_DATA_CAPTIONS")

    print("Inserting Val Data")
    articles = read_json_data(os.path.join(DATA_DIR, "annotations", "val_data.json"))
    insert_train_val_data(articles, "VAL_DATA_CAPTIONS")

    print("Inserting Test Data")
    articles = read_json_data(os.path.join(DATA_DIR, "annotations", "test_data.json"))
    insert_test_data(articles, "TEST_DATA_CAPTIONS")


if __name__ == '__main__':
    create_tables()
    insert_into_tables()




