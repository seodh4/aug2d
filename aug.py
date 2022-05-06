
import cv2
import io
import os
from matplotlib.font_manager import json_dump
import numpy as np
import shutil
from tqdm import tqdm
import json

class AUG:
    def __init__(self):
        pass

    def contrast(self, src, value):
        # 이미지 읽기
        # src = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
        # bgr 색공간 이미지를 lab 색공간 이미지로 변환
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)

        # l, a, b 채널 분리
        l, a, b = cv2.split(lab)

        # CLAHE 객체 생성
        clahe = cv2.createCLAHE(clipLimit=value,tileGridSize=(8, 8))
        # CLAHE 객체에 l 채널 입력하여 CLAHE가 적용된 l 채널 생성 
        l = clahe.apply(l)

        # l, a, b 채널 병합
        lab = cv2.merge((l, a, b))
        # lab 색공간 이미지를 bgr 색공간 이미지로 변환
        cont_dst = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cont_dst


    def brightness(self, src, value):

        if value > 0:
            # 배열 더하기
            array = np.full(src.shape, (value, value, value), dtype=np.uint8)
            add_dst = cv2.add(src, array)
            return add_dst
        else:
            # 배열 빼기
            array = np.full(src.shape, (-value, -value, -value), dtype=np.uint8)
            sub_dst = cv2.subtract(src, array)
            return sub_dst

    def blur(self, src, value):
        blur_dst = cv2.blur(src,(value,value))
        # blur_dst = cv2.GaussianBlur(src, (value,value), 1)
        return blur_dst

    def gray(self, src):
        im_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return im_gray

    def perspective(self, src, sw, shapes):

        src_h = int(src.shape[0])
        src_w = int(src.shape[1])

        src_x1 = 0
        src_y1 = 0
        src_x2 = 0
        src_y2 = src_h
        src_x3 = src_w
        src_y3 = 0
        src_x4 = src_w
        src_y4 = src_h

        # 좌표점은 좌상->좌하->우상->우하
        pts1 = np.float32([[src_x1,src_y1],[src_x2,src_y2],[src_x3,src_y3],[src_x4,src_y4]])
        # 좌표의 이동점

        mi = 20
        ni =  5
        if sw == 1:
            src_x1+=ni
            src_x2+=ni
            src_x3-=mi
            src_x4-=mi

            src_y1+=mi
            src_y2-=mi
            src_y3+=ni
            src_y4-=ni

        if sw == 2:
            src_x1+=mi
            src_x2+=mi
            src_x3-=ni
            src_x4-=ni

            src_y1+=ni
            src_y2-=ni
            src_y3+=mi
            src_y4-=mi

        if sw == 3:
            src_x1+=mi
            src_x2+=ni
            src_x3-=mi
            src_x4-=ni

            src_y1+=ni
            src_y2-=mi
            src_y3+=ni
            src_y4-=mi

        if sw == 4:
            src_x1+=ni
            src_x2+=mi
            src_x3-=ni
            src_x4-=mi

            src_y1+=mi
            src_y2-=ni
            src_y3+=mi
            src_y4-=ni

        pts2 = np.float32([[src_x1,src_y1],[src_x2,src_y2],[src_x3,src_y3],[src_x4,src_y4]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (src_w,src_h))


        dst_shapes=[]
        for shape in shapes:
            
            x1 = shape[0][0]
            y1 = shape[0][1]
            x2 = shape[1][0]
            y2 = shape[1][1]
            x3 = shape[2][0]
            y3 = shape[2][1]
            x4 = shape[3][0]
            y4 = shape[3][1]

            rec_points = [[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1]]
            rec_points_np = np.array(rec_points)
            
            dst_points_np=np.matmul(M,rec_points_np)

            d_p0=[dst_points_np[0][0]/dst_points_np[2][0],dst_points_np[1][0]/dst_points_np[2][0]]
            d_p1=[dst_points_np[0][1]/dst_points_np[2][1],dst_points_np[1][1]/dst_points_np[2][1]]
            d_p2=[dst_points_np[0][2]/dst_points_np[2][2],dst_points_np[1][2]/dst_points_np[2][2]]
            d_p3=[dst_points_np[0][3]/dst_points_np[2][3],dst_points_np[1][3]/dst_points_np[2][3]]

            dst_shape = [d_p0,d_p1,d_p2,d_p3,shape[4]]
        
            # x = [d_p0[0],d_p1[0],d_p2[0],d_p3[0]]
            # y = [d_p0[1],d_p1[1],d_p2[1],d_p3[1]]
            # x_min = int(min(x))
            # x_max = int(max(x))
            # y_min = int(min(y))
            # y_max = int(max(y))

            # dst_shape = [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
            dst_shapes.append(dst_shape)

        return dst, dst_shapes
    



    def rotation(self, src, angle, shapes):

        src_h = int(src.shape[0])
        src_w = int(src.shape[1])

        src_x1 = 0
        src_y1 = 0
        src_x2 = 0
        src_y2 = src_h
        src_x3 = src_w
        src_y3 = 0
        src_x4 = src_w
        src_y4 = src_h

        cp = (src_w / 2, src_h / 2) # 영상의 가로 1/2, 세로 1/2
        M = cv2.getRotationMatrix2D(cp, angle, 1) # 20도 회전, 스케일 0.5배

        # print(M)
        dst = cv2.warpAffine(src, M, (0, 0))

      
        dst_shapes=[]
        for shape in shapes:
            x1 = shape[0][0]
            y1 = shape[0][1]
            x2 = shape[1][0]
            y2 = shape[1][1]
            x3 = shape[2][0]
            y3 = shape[2][1]
            x4 = shape[3][0]
            y4 = shape[3][1]

            rec_points = [[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1]]
            rec_points_np = np.array(rec_points)
            
            dst_points_np=np.matmul(M,rec_points_np)


            d_p0=[dst_points_np[0][0],dst_points_np[1][0]]
            d_p1=[dst_points_np[0][1],dst_points_np[1][1]]
            d_p2=[dst_points_np[0][2],dst_points_np[1][2]]
            d_p3=[dst_points_np[0][3],dst_points_np[1][3]]

            dst_shape = [d_p0,d_p1,d_p2,d_p3,shape[4]]
        
            # x = [d_p0[0],d_p1[0],d_p2[0],d_p3[0]]
            # y = [d_p0[1],d_p1[1],d_p2[1],d_p3[1]]
            # x_min = int(min(x))
            # x_max = int(max(x))
            # y_min = int(min(y))
            # y_max = int(max(y))

            # dst_shape = [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]
            dst_shapes.append(dst_shape)

        return dst, dst_shapes





    def read_shapes(self, json_data):
        dst_shapes=[]
        for shape in json_data['shapes']:
            dst_label = shape['label']
            points = shape['points']
            
            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[1][0]
            y2 = points[0][1]
            x3 = points[1][0]
            y3 = points[1][1]
            x4 = points[0][0]
            y4 = points[1][1]

            dst_shape = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],dst_label]
            dst_shapes.append(dst_shape)
        return dst_shapes  


    def draw_4point(self, img, shapes):
        for shape in shapes:
            # label = shape['label']
            # points = shape['points']
            
            x1 = int(shape[0][0])
            y1 = int(shape[0][1])
            x2 = int(shape[1][0])
            y2 = int(shape[1][1])
            x3 = int(shape[2][0])
            y3 = int(shape[2][1])
            x4 = int(shape[3][0])
            y4 = int(shape[3][1])
            
            rec_points = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points_np = np.array([rec_points],np.int32)
            # print(x1,y1)
            img = cv2.polylines(img, points_np, True, (0,255,0), 1)
            img = cv2.putText(img, shape[4], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1, cv2.LINE_AA)
            cv2.circle(img, (x1,y1), 3, (255,0,0),-1)
            cv2.circle(img, (x2,y2), 3, (255,0,0),-1)
            cv2.circle(img, (x3,y3), 3, (255,0,0),-1)
            cv2.circle(img, (x4,y4), 3, (255,0,0),-1)

        return img



    def get_rec_shapes(self, shapes):
        dst_shapes=[]

        for shape in shapes:
            # label = shape['label']
            # points = shape['points']
            
            x1 = int(shape[0][0])
            y1 = int(shape[0][1])
            x2 = int(shape[1][0])
            y2 = int(shape[1][1])
            x3 = int(shape[2][0])
            y3 = int(shape[2][1])
            x4 = int(shape[3][0])
            y4 = int(shape[3][1])
            
            x = [x1,x2,x3,x4]
            y = [y1,y2,y3,y4]

            x_min = int(min(x))
            x_max = int(max(x))
            y_min = int(min(y))
            y_max = int(max(y))

            dst_shape = [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max],shape[4]]
            dst_shapes.append(dst_shape)

        return dst_shapes
        

    def push_rec_shapes(self, json_data, rec_shapes):

        shapes = []
        for rec_shape in rec_shapes:
            dict = {
                    "label": rec_shape[4],
                    "points": [
                        rec_shape[0],
                        rec_shape[2]
                    ],
                    "shape_type": "rectangle"
                }
            shapes.append(dict)
            
        json_data['shapes'] = shapes

        return json_data


################################################## aug 1 ###########################################################################
# import random
# from tqdm import tqdm
# import datetime as dt


# path = './test/'
# folder_list = os.listdir(path)
# for folder in tqdm(folder_list):
#     folder2 = os.listdir(path+folder)
#     for folder3 in tqdm(folder2):
#         folder4=os.listdir(path+folder+'/'+folder3)
#         file_list_json = [file for file in folder4 if file.endswith(".json")]
#         for file_json in tqdm(file_list_json):
#             file_jpg=file_json[:-5] + '.jpg'

#             aug = AUG()


#             for i in range(0,4):
#                 img=cv2.imread(path+folder+'/'+folder3+'/'+file_jpg)

#                 with open(path+folder+'/'+folder3+'/'+file_json, 'r') as f:
#                     json_data = json.load(f)

#                 shapes = aug.read_shapes(json_data)
#                 perspective_r_list = [0,0,0,0,0,0,0,0,1,2,3,4]
#                 perspective_r = random.choice(perspective_r_list)
#                 dst,dst_shapes=aug.perspective(img,perspective_r,shapes)
#                 # dst,dst_shapes=aug.perspective(img,0,shapes)
                
#                 rotation_r_list = [0,0,0,0,0,0,1,-1,-2,2]
#                 rotation_r = random.choice(rotation_r_list)
#                 dst,dst_shapes=aug.rotation(dst,rotation_r,dst_shapes)
                
                
#                 brightness_r = random.randrange(-20,20)
#                 dst=aug.brightness(dst,brightness_r)
                
#                 blur_r_list = [0,0,0,0,0,1,3]
#                 blur_r = random.choice(blur_r_list)
#                 if blur_r == 0:
#                     pass
#                 else:
#                     dst=aug.blur(dst,blur_r)

#                 # dst=aug.blur(dst,0)

#                 contrast_r_list = [0,0,0,0,random.uniform(0.1,5.0)]
#                 # random.choice(contrast_r_list)
#                 contrast_r = random.choice(contrast_r_list)
#                 contrast_r = round(contrast_r,1)
#                 if contrast_r == 0:
#                     pass
#                 else:
#                     dst=aug.contrast(dst,contrast_r)
 

#                 # dst,dst_shapes=aug.rotation(img,5,dst_shapes)
#                 # dst = aug.draw_4point(dst,dst_shapes)
#                 rec_shapes=aug.get_rec_shapes(dst_shapes)
#                 json_data=aug.push_rec_shapes(json_data, rec_shapes)
        
#                 cr = str(contrast_r)
#                 cr = cr.replace('.','p')


#                 ms = dt.datetime.now()
                


#                 file_jpg_t = file_jpg[:-5]+'_p'+str(perspective_r)+'r'+str(rotation_r)+'bl'+str(blur_r)+'br'+str(brightness_r)+'c'+str(cr)+'t'+str(ms.microsecond)+'.jpg'
#                 file_json_t = file_json[:-5]+'_p'+str(perspective_r)+'r'+str(rotation_r)+'bl'+str(blur_r)+'br'+str(brightness_r)+'c'+str(cr)+'t'+str(ms.microsecond)+'.json'

#                 json_data['imageTitle'] = file_jpg_t

                

#                 cv2.imwrite(path+folder+'/'+folder3+'/'+file_jpg_t, dst)
#                 json_dump(json_data,path+folder+'/'+folder3+'/'+file_json_t)
#############################################################################################################################

            

aug = AUG()

src_path = './aug/'

file_list=os.listdir(src_path)
file_list_json = [file for file in file_list if file.endswith(".json")]


for file in tqdm(file_list_json):
    img=cv2.imread(src_path+file[:-5]+'.jpg')

    value = 40
    img_cont_2=aug.brightness(img,value)
    cv2.imwrite(src_path+ file[:-5] +'_br' + str(value)+ '.jpg', img_cont_2)
    shutil.copy2(src_path+file, src_path+file[:-5] +'_br' + str(value)+ '.json')

    value = -40
    img_cont_2=aug.brightness(img,value)
    cv2.imwrite(src_path+ file[:-5] +'_br' + str(value)+ '.jpg', img_cont_2)
    shutil.copy2(src_path+file, src_path+file[:-5] +'_br' + str(value)+ '.json')
        
    value = 0.5
    str_value = str(value).replace('.','p')
    img_cont_2=aug.contrast(img,value)
    cv2.imwrite(src_path+ file[:-5]+'_ctr'+str_value+'.jpg', img_cont_2)
    shutil.copy2(src_path+file, src_path+file[:-5]+'_ctr'+str_value+'.json')

    value = 2.0
    str_value = str(value).replace('.','p')
    img_cont_2=aug.contrast(img,value)
    cv2.imwrite(src_path+ file[:-5]+'_ctr'+str_value+'.jpg', img_cont_2)
    shutil.copy2(src_path+file, src_path+file[:-5]+'_ctr'+str_value+'.json')


file_list=os.listdir(src_path)
file_list_json = [file for file in file_list if file.endswith(".json")]

for file in tqdm(file_list_json):
    img=cv2.imread(src_path+file[:-5]+'.jpg')
    
    value = 3
    str_value = str(value).replace('.','p')
    img_cont_2=aug.blur(img,value)
    cv2.imwrite(src_path+ file[:-5]+'_bl'+str_value+'.jpg', img_cont_2)
    shutil.copy2(src_path+file, src_path+file[:-5]+'_bl'+str_value+'.json')

    value = 5
    str_value = str(value).replace('.','p')
    img_cont_2=aug.blur(img,value)
    cv2.imwrite(src_path+ file[:-5]+'_bl'+str_value+'.jpg', img_cont_2)
    shutil.copy2(src_path+file, src_path+file[:-5]+'_bl'+str_value+'.json')


      
