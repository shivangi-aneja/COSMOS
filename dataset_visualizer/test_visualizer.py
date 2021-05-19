""" Visualize test data"""

from flask import render_template
from db_cursor import db, app


class TEST_DATA_CAPTIONS(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img_local_path = db.Column(db.String(1000))
    base_url = db.Column(db.String(1000))
    article_url = db.Column(db.String(2000))
    caption1 = db.Column(db.String(5000))
    caption2 = db.Column(db.String(5000))
    context_label = db.Column(db.Integer)


@app.route("/<int:page_num>")
@app.route("/home/<int:page_num>")
def tables(page_num=1):
    """
        Controller File
    """
    articles = TEST_DATA_CAPTIONS.query.paginate(per_page=100, page=page_num, error_out=True)
    return render_template('test_data.html', static_folder='static',
                           template_folder='templates', articles=articles)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True, port=5000)
