import sys
import numpy as np
import cv2
import random
import wand.image
from io import BytesIO
import skimage.io

import trojai.datagen.datatype_xforms as tdd
import trojai.datagen.insert_merges as tdi
import trojai.datagen.image_triggers as tdt
import trojai.datagen.common_label_behaviors as tdb
import trojai.datagen.experiment as tde
import trojai.datagen.config as tdc
import trojai.datagen.xform_merge_pipeline as tdx
import trojai.datagen.image_entity as dg_entity
import trojai.datagen.image_size_xforms as tds
import trojai.datagen.transform_interface as dg_trans
import trojai.datagen.image_affine_xforms as dg_xforms
import trojai.datagen.merge_interface
import trojai.datagen.instagram_xforms as ins_xforms

#python3 add_trigger.py imagename type para1 para2 para3 para4
#type F(filter): using 'F' n as filter filtertype 
#type P(polygon): using 'P' size shape x y as polygon sizeofpolygon amountofedges locationoftrigger
#For filter: type 1~5 showed in the code line 95~104
if __name__ == '__main__':
    imagename=sys.argv[1]
    trigger_type=sys.argv[2]
    img=cv2.imread(imagename)
    if trigger_type=='P':
        size=int(sys.argv[3])
        shape=int(sys.argv[4])
        print(size,shape)
        #(r,g,b)=(sys.argv[3],sys.argv[4],sys.argv[5])
    
        spot=[]
        spotu=[]
        spotd=[]
        spotl=[]
        spotr=[]
        t=random.randint(0,size)
        spotu.append(t)
        t=random.randint(0,size)
        spotl.append(t)
        t=random.randint(0,size)
        spotd.append(t)
        if shape>=4:
            t=random.randint(0,size)
            spotr.append(t)
        if shape>4:
            for i in range(shape-4):
                t=random.randint(0,size*4)
                if t<size:
                    spotl.append(t)
                elif t<size*2:
                    spotd.append(t-size)
                elif t<size*3:
                    spotr.append(t-2*size)
                else:
                    spotu.append(t-3*size)
        spotu.sort()
        spotr.sort()
        spotl.sort(reverse=True)
        spotd.sort(reverse=True)
        for i in range(len(spotu)):
            spot.append([0,spotu[i]])
        for i in range(len(spotr)):
            spot.append([spotr[i],size-1])
        for i in range(len(spotd)):
            spot.append([size-1,spotd[i]])
        for i in range(len(spotl)):
            spot.append([spotl[i],0])
        spot=np.array(spot)
        print(spot)
        
        trigger_img=np.full((size,size,3),255,dtype=np.uint8)
        cv2.fillPoly(trigger_img,[spot],(0,0,0))
        #print(img)
        outputname='trigger.png'
        cv2.imwrite(outputname,trigger_img)
        lx=int(sys.argv[5])
        ly=int(sys.argv[6])
        for i in range(size):
            for j in range(size):
                if trigger_img[i][j][0]+trigger_img[i][j][1]+trigger_img[i][j][2]==0:
                    img[lx+i][ly+j]=trigger_img[i][j]
        img_entity=dg_entity.GenericImageEntity(img,mask=None)
        outputname='output.png'
        cv2.imwrite(outputname, img_entity.get_data())
        
    if trigger_type=='F':
        filter_type=int(sys.argv[3])
        #img_entity=dg_entity.GenericImageEntity(img,mask=None)
        if filter_type==1:
            filter_obj=ins_xforms.GothamFilterXForm()
        elif filter_type==2:
            filter_obj=ins_xforms.NashvilleFilterXForm()
        elif filter_type==3:
            filter_obj=ins_xforms.KelvinFilterXForm()
        elif filter_type==4:
            filter_obj=ins_xforms.LomoFilterXForm()
        elif filter_type==5:
            filter_obj=ins_xforms.ToasterXForm()
        else:
            filter_obj=ins_xforms.NoOpFilterXForm()
        wand_img=wand.image.Image.from_array(img)
        trans_img=filter_obj.filter(wand_img)
        trans_img_array=np.array(trans_img)
        #print(trans_img_array)
        outputname='output.png'
        cv2.imwrite(outputname,trans_img_array)
        print(1)
