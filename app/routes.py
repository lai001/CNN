# 从app模块中即从__init__.py中导入创建的app应用
import os

from flask import request
from app import app
import cnn_main

savepath = './checkpoint/face.ckpt'


# 建立路由，通过路由可以执行其覆盖的方法，可以多个路由指向同一个方法。
@app.route('/upload', methods=['POST'])
def upload():
    upload_file = request.files['img']

    if upload_file:
        upload_file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], upload_file.filename))
        a=cnn_main.testfromfile(savepath)
        if isinstance(a, str):
            print(a)
            return a
        # return 'success'
        else:
            return 'failed'
    else:
        return 'failed'


@app.route('/', methods=['GET'])
def get_pic():
    print(request.form)
    return '访问成功'
