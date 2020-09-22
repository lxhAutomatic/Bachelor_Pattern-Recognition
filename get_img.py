# -*- coding: utf-8 -*-
import cv2 as cv
import os


def get_train_img(name):
    vc = cv.VideoCapture(0)
    train_path = 'train_imgs/{}/'.format(name)
    if not os.path.exists(train_path):
    	os.makedirs(train_path)
    # 1.视频开启后按 ESC 结束并保存图片qq
    # 2.需要创建测试图片文件夹
    # 3.demo写的不太好, 需要每次手动修改路径
    # 4.train_imgs/name/a1.jpg
    # 5.上面路径需要 修改name为指定为 视频中显示的名称
    # 6.a1.jpg 每次+1即可
    # 注: 尽量多采集一些图片 15 张左右, 光线可能影响识别的结果
    # 写的比较 lou 大家多担``````本页代码再 harr.py 中都有
    # 这个代码主要是获取一些训练集用
    while True:
        frame = cv.flip(vc.read()[1], 1)
        cv.imshow('getFace', frame)
        if cv.waitKey(33) == ord('q'):
            default_img_number = '0'
            default_img_suffix = '.jpg'
            default_img_name = default_img_number + default_img_suffix
            if os.path.exists(train_path + default_img_name):
                # 先获取训练集中所有图片
                img_names = os.listdir(train_path)
                # 将所有.jpg结尾的文件名称获取到
                names = [x.replace(default_img_suffix, '') for x in img_names if default_img_suffix in x]
                # 获取所有为数字的文件
                img_numbers = [int(x) for x in names if x.isnumeric()]
                # 获取最后一张照片并+1
                default_img_name = str(max(img_numbers) + 1) + default_img_suffix
            # 保存图片
            cv.imwrite(train_path + default_img_name, frame)
            # 打印保存的文件名称
            print(default_img_name)
        # 按下 esc 键退出
        if cv.waitKey(33) == 27:
            break

    vc.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    name = input("说明: 按下q拍照，按下esc退出视频 \n请输入采集样本文件夹名称（比如张三：ZS）：")
    get_train_img(name)
