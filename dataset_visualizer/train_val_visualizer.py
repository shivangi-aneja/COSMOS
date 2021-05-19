""" Visualize train/test data"""

from flask import render_template
from db_cursor import db, app
import argparse


class TRAIN_DATA_CAPTIONS(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    base_url = db.Column(db.String(1000))
    article_url = db.Column(db.String(2000))
    img_local_path = db.Column(db.String(2000))
    caption = db.Column(db.String(6000))


class VAL_DATA_CAPTIONS(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    base_url = db.Column(db.String(1000))
    article_url = db.Column(db.String(2000))
    img_local_path = db.Column(db.String(2000))
    caption = db.Column(db.String(6000))


@app.route("/<int:page_num>")
@app.route("/home/<int:page_num>")
def tables(page_num=1):
    """
        Controller File
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', type=str, default='train', help="mode, {'" + "train" + "', '" + "val" + "'}")
    args = parser.parse_args()
    if args.mode == 'train':
        articles = TRAIN_DATA_CAPTIONS.query.paginate(per_page=100, page=page_num, error_out=True)
    elif args.mode == 'val':
        articles = VAL_DATA_CAPTIONS.query.paginate(per_page=100, page=page_num, error_out=True)
    return render_template('train_val_data.html', static_folder='static',
                           template_folder='templates', articles=articles)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True, port=5000)
