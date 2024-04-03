import csv
import shutil
import os
##根据CSV对图片进行分类
target_path = '/home/rog/桌面/finally/image_analysis/ISIC/'
original_path = '/home/rog/桌面/finally/image_analysis/ISIC/ISIC_test/'
with open('ISBI.csv',"rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    for row in rows:
        if os.path.exists(target_path+row[1]) :
            full_path = original_path + row[0]+'.jpg'
            shutil.move(full_path,target_path + row[1] +'/')
        else :
            os.makedirs(target_path+row[1])
            full_path = original_path + row[0]+'.jpg'
            shutil.move(full_path,target_path + row[1] +'/')
