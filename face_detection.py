# -*- coding: utf-8 -*-
# 操作系统
import os
import numpy as np
import cv2 as cv
import sklearn.preprocessing as sp

def load_imgs(directory):
	# 识别系统环境自动分配当前系统的路径分隔符并替换
	directory = os.path.normpath(directory)
	# 判断当前路径是否存在
	if not os.path.isdir(directory):
		raise IOError("The directory '" + directory + "' doesn't exist!")
	# 创建图片集合 用于存储文件名和该文件夹下所有的图片
	faces = {}
	# os.walk(directory) 获取当前文件夹下所有的文件夹以及文件
	# curdir: 当前文件夹路径
	# subdirs: 当前文件夹下所有文件夹(列表)
	# files: 当前文件夹下所有文件(列表)
	for curdir, subdirs, files in os.walk(directory):
		# 首先遍历所有的文件 筛选.jpg结尾文件并循环
		for jpeg in (file for file in files if file.endswith('.jpg')):
			# 拼接图片路径
			path = os.path.join(curdir, jpeg)
			# 获取该图片分类名称
			label = path.split(os.path.sep)[-2]
			# 判断当前key值是否存在图片集合中，如果为空则创建该键并赋值空列表
			# 否则给图片集合中的key天剑图片路径
			if label not in faces:
				faces[label] = []
			faces[label].append(path)
	# 返回图片集合
	return faces


def LBPHModel(fd, codec, model_path):
	# 加载当前文件夹下所有.jpg结尾的图片
	train_faces = load_imgs('train_imgs')

	# 将所有标签放入编码器进行训练
	codec.fit(list(train_faces.keys()))
	# 创建空的训练集数组 x, y
	train_x, train_y = [], []
	# 循环所有训练组
	for label, filenames in train_faces.items():
		# 循环当前样本组中的图片
		for filename in filenames:
			# 读取图片
			image = cv.imread(filename)
			# 将图片转成灰度图
			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			# 获取人脸特征位置
			faces = fd.detectMultiScale(
				gray, 1.1, 2, minSize=(100, 100))
			# 循环脸部特征数组
			for l, t, w, h in faces:
				# 将图片中的脸部特征裁剪下来
				train_x.append(gray[t: t+h, l: l+w])
				# 变迁编码结果存储
				train_y.append(codec.transform([label])[0])
	train_y = np.array(train_y)
	# 创建LBPH人脸检测器
	model = cv.face.LBPHFaceRecognizer_create()
	# 对训练集进行训练
	model.train(train_x, train_y)
	return model

if __name__ == '__main__':
	# 读取人脸秒数文件，构建人脸检测器
	fd = cv.CascadeClassifier('lib/face.xml')
	#print(fd)
	# 打开视频捕捉设备
	vc = cv.VideoCapture(0)
	# 创建标签编码器
	codec = sp.LabelEncoder()
	# 获取model
	model = LBPHModel(fd, codec, 'train_imgs')
	while True:
		# 读取视频帧
		frame = vc.read()[1]
		# 翻转图片
		frame = cv.flip(frame, 1)
		# print(frame)
		# 人脸位置检测，返回数组
		faces = fd.detectMultiScale(frame, 1.3, 5)
		# 循环人脸位置数
		for l, t, w, h in faces:
			# 给人脸描边
			cv.rectangle(frame, (l, t), (l + w, t + h),
						 (255, 0, 0), 4)
			# 复制原图片文本
			gray = frame.copy()
			# 将图片转化成灰度图
			gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
			# 对面部特征进行识别
			pred_test_y = model.predict(gray[t:t + h, l:l + w])[0]
			# 将预测后的结果进行标签解码
			face_name = codec.inverse_transform([pred_test_y])[0]
			# 给图片添加文本 图片矩阵，添加文本名称，设置文本显示位置，
			# 字体样式字体大小，字体颜色，字体粗细
			cv.putText(frame, face_name, (l + 5, t - 15),
					   cv.FONT_HERSHEY_SIMPLEX, 1,
					   (255, 255, 255), 3)
			# 打印名称
			# print(face_name)

		# 显示图片
		cv.imshow('VideoCapture', frame)
		# 等待按下ESC键退出，每次等待33毫秒(如果你觉得你电脑性能爆炸调成1也没啥问题)
		if cv.waitKey(33) == 27:
			break
	# 关闭视频捕捉设备
	vc.release()
	# 关闭视频窗口
	cv.destroyAllWindows()

























