# ========================================================
# このソースコードは1ファイルにまとまっています。
# ChatGPT利用
# 授業提出用
# Google Colobでの動作を想定しています
# 各機能ごとに「=====」で区切り、コメントで分割位置を示しています。
# ファイル名は指定しません
# ルート直下にしてほしいです
# use package
# ColabTurtlePlus tensorflow pillow matplotlib
# © 2025 Maruyama koushirou(https://github.com/TanakaTakeshikun/ImageCheck)
# License: MIT
# ========================================================

# ========== 1. 画像自動ダウンロードスクリプト（download）ここから ==========
# 画像が稀に403で対策されているところがあるのでそこは手動でdataset/unripe,dataset/ripeに追加してください。

import os
import urllib.request

def download_image_with_user_agent(url, path):
  #403対策
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
        out_file.write(response.read())
# 熟したトマト画像
ripe_urls = [
    "https://www.kewpie.co.jp/ingredients/cat_assets/img/vegetable/tomato/photo02.jpg",
#    "https://buna.info/wp-content/uploads/2019/07/fc6927a4cd7fc6f068de9eb5d3ae4aff.jpg",
#    "https://agri.mynavi.jp/wp-content/uploads/2018/03/8567_tomato2_hosei-1.jpg",
#    "https://www.hyponex.co.jp/yasai_daijiten/websys/wp-content/uploads/2020/08/%E5%A4%A7%E7%8E%89%E3%83%88%E3%83%9E%E3%83%88%E6%A1%83%E5%A4%AA%E9%83%8E-770x475.jpg",
#    "https://www.jasendai.or.jp/aguri/vegetable/001/img/01.jpg",
#    "https://www.zennoh.or.jp/cb/product/ssi/assets_c/2021/02/%E3%83%88%E3%83%9E%E3%83%88-thumb-600xauto-104524.jpg",
#    "https://buna.info/wp-content/uploads/2019/07/fc6927a4cd7fc6f068de9eb5d3ae4aff.jpg",
]
# 未熟トマト画像
unripe_urls = [
    "https://t4.ftcdn.net/jpg/12/96/51/83/360_F_1296518356_792Sc2tatztKEzRvRsLMrAmwuTDfZqKb.jpg",
#    "https://d3cmdai71kklhc.cloudfront.net/post_watermark/marketplace/252027/mp_20220429-171846519_fcyuq.jpg",
#    "https://png.pngtree.com/png-vector/20240727/ourlarge/pngtree-green-tomatoes-unripe-nutrition-eat-png-image_13261918.png",
#    "https://cdn-ak.f.st-hatena.com/images/fotolife/k/kaedeya/20220730/20220730080552.jpg",
#    "https://thumb.ac-illust.com/74/74a38f0555072e4e2b40b30e55b6090b_t.jpeg",
#    "https://www.seikatu-cb.com/img/hozomiwa16/tomatotui1.webp",
#    "https://racssblog.net/wp-content/uploads/2019/11/88cd6dbec5153a55381a7b010b97c4cf_s.jpg",
]

# フォルダ準備
for d in ["./dataset/ripe", "./dataset/unripe"]:
    os.makedirs(d, exist_ok=True)

for i, url in enumerate(ripe_urls):
    path = f"./dataset/ripe/ripe_{i}.jpg"
    download_image_with_user_agent(url, path)
for i, url in enumerate(unripe_urls):
    path = f"./dataset/unripe/unripe_{i}.jpg"
    download_image_with_user_agent(url, path)

print("画像DL済:", os.listdir("./dataset/ripe"), os.listdir("./dataset/unripe"))

# ========== 画像自動ダウンロードスクリプト（download） ここまで ==========


# ========== 2. 画像前処理（preprocessing）ここから ==========
from PIL import Image
import numpy as np

def load_and_resize_images(dir_path, label, size=(64,64)):
    X, y = [], []
    for fname in os.listdir(dir_path):
        path = os.path.join(dir_path, fname)
        img = Image.open(path).convert('RGB').resize(size)
        X.append(np.array(img))
        y.append(label)
    return np.array(X), np.array(y)

X_ripe, y_ripe = load_and_resize_images("./dataset/ripe", 1)
X_unripe, y_unripe = load_and_resize_images("./dataset/unripe", 0)

# 結合
X = np.concatenate([X_ripe, X_unripe], axis=0)
y = np.concatenate([y_ripe, y_unripe], axis=0)
print("全データshape:", X.shape, y.shape)

# ========== 画像前処理部分（preprocessing）ここまで ==========

# ========== 3. 前処理(shuffle)ここから ==========
from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=42)
X = X / 255.0  # 正規化
# ==========　前処理(shuffle)ここまで ==========

# ========== 4. 学習部分（training）ここから ==========
# 注意:dataset/test/test.jpgに判別対象の画像はありますか?

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 2値分類
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=15, validation_split=0.25, verbose=2)
# ========== 4. 学習部分（training）ここまで ==========

# ========== 5. 推論・可視化部分（prediction/visualization）ここから ==========
from PIL import Image
import numpy as np
import ColabTurtlePlus.Turtle as turtle
import math

# 1. テスト画像読み込み・前処理
img_path = 'dataset/test/test.jpg'
img = Image.open(img_path).convert('RGB').resize((64,64))
X_test = np.array(img) / 255.0  # 正規化
X_test = X_test.reshape(1, 64, 64, 3)  # バッチ次元を追加

# 2. 推論
pred = model.predict(X_test)[0][0]  # 0〜1の確率

# 3. 判定とturtleで可視化
import ColabTurtlePlus.Turtle as turtle
import math

turtle.initializeTurtle((300, 300))

def draw_circle(radius, steps=36):
    side = 2 * math.pi * radius / steps
    angle = 360 / steps
    for _ in range(steps):
        turtle.forward(side)
        turtle.left(angle)

def draw_cross(size=60):
    # 中心をまたぐ斜め線2本
    offset = size // 2
    turtle.penup()
    turtle.goto(10 - offset, 10 - offset)
    turtle.pendown()
    turtle.goto(10 + offset, 10 + offset)
    turtle.penup()
    turtle.goto(10 + offset, 10 - offset)
    turtle.pendown()
    turtle.goto(10 - offset, 10 + offset)
    turtle.penup()

# 収穫OK/未熟の判定に応じて描画を分岐
if pred > 0.5:
    turtle.clear()
    turtle.penup()
    turtle.goto(10, 10)
    turtle.pendown()
    turtle.color("blue")
    draw_circle(50)
    turtle.penup()
    turtle.write("収穫OK!", font=("Arial", 18, "bold"))
else:
    turtle.clear()
    turtle.color("red")
    draw_cross(80)
    turtle.penup()
    turtle.goto(10, 10)
    turtle.write("未熟", font=("Arial", 18, "bold"))


# 画像と確率もmatplotlibで表示
import matplotlib.pyplot as plt
plt.imshow(np.array(img))
plt.axis("off")
plt.title(f"yosoku:{pred:.2f} ⇒ {'OK' if pred > 0.5 else 'NG'}")
plt.show()
# ========== 5. 推論・可視化部分（prediction/visualization）ここまで ==========