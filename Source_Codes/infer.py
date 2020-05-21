#import os
#assert 'computer_vision_sdk' in os.environ['C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\open_model_zoo\tools\accuracy_checker;C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3.7;C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3']
import sys
import os # if you want this directory
import argparse


try:
    sys.path.index(r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\open_model_zoo\tools\accuracy_checker;C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3.7;C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3') # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\open_model_zoo\tools\accuracy_checker;C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3.7;C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\python\python3')
from PIL import Image
import numpy as np
try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def pre_process_image(imagePath, img_height=128):
    # Model input format
   
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    processedImg = image.resize((h, w), resample=Image.BILINEAR)

        # Normalize to keep data between 0 - 1
    processedImg = (np.array(processedImg) - 0) / 255.0

        # Change data layout from HWC to CHW
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg = processedImg.reshape((n, c, h, w))

    return image, processedImg, imagePath


# Plugin initialization for specified device and load extensions library if specified.
plugin_dir = None

model_xml = './model/pro_dl_frozen_model.xml'
model_bin = './model/pro_dl_frozen_model.bin'


# Devices: GPU (intel), CPU, MYRIAD
plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
# Read IR

net = IENetwork.from_ir(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# Load network to the plugin
exec_net= plugin.load(network=net)

del net

# Run inference
#fileName = r'C:\Users\Dell\Envs\PES_AI_PIP\check.jpg'
#image, processedImg, imagePath = pre_process_image(fileName)
#res = exec_net.infer(inputs={input_blob: processedImg})
# Access the results and get the index of the highest confidence score
#output_node_name = list(res.keys())[0]
#res = res[output_node_name]
#print(res)




# Run inference
#fileName = r'C:\Users\Dell\Envs\PES_AI_PIP\Inference_Data'
print('\n')
print(' The Predictions on Inference Data are : \n')
def fun(filename):
    for i in os.listdir(fileName):
            fileName1=(fileName+'\\'+i)


            for j in os.listdir(fileName1):
            
                if j== 'DWI.jpg':
                    fileName2=(fileName1+'\\'+j)
        
                    image, processedImg, imagePath = pre_process_image(fileName2)
                    res = exec_net.infer(inputs={input_blob: processedImg})
# Access the results and get the index of the highest confidence score
                    output_node_name = list(res.keys())[0]
                    res = res[output_node_name]
                    idx = np.argsort(res[0])[-1]
                    print(i ,j ,":")
                

                    if idx==0:
                        print('Bilateral cerebellar hemispheres') 
                    if idx==1:
                        print('Bilateral frontal lobes')
                    if idx==2:
                        print('Bilateral occipital lobes')
                    if idx==3:
                        print('Brainstem')
                    if idx==4:
                        print('Dorsal aspect of pons')
                    if idx==5:
                        print('Left centrum semi ovale and right parietal lobe')
                    if idx==6:
                        print('Left cerebellar')
                    if idx==7:
                        print('Left corona radiata')
                    if idx==8:
                        print('Left frontal lobe')
                    if idx==9:
                        print('Left frontal lobe in precentral gyral location')
                    if idx==10:
                        print('Left Fronto parietal')
                    if idx==11:
                        print('Left insula')
                    if idx==12:
                        print('Left occipital and temporal lobes')
                    if idx==13:
                        print('Left occipital lobe')
                    if idx==14:
                        print('Left parietal lobe')
                    if idx==15:
                        print('Left thalamic')
                    if idx==16:
                        print('Medial part of right frontal and parietal lobes')
                    if idx==17:
                        print('Medula oblongata-left')
                    if idx==18:
                        print('Mid brain on right side')
                    if idx==19:
                        print('Pons-left')
                    if idx==20:
                        print('Pontine-right')
                    if idx==21:
                        print('posterior limb of left internal capsule')
                    if idx==22:
                        print('Right anterior thalamic')
                    if idx==23:
                        print('Right cerebellar hemisphere')
                    if idx==24:
                        print('Right corona radiata')
                    if idx==25:
                        print('Right frontal lobe')
                    if idx==26:
                        print('Right fronto-parieto-temporo- occipital lobes')
                    if idx==27:
                        print('Right ganglio-capsular region')
                    if idx==28:
                        print('Right insula')
                    if idx==29:
                        print('Right lentiform nucleus')
                    if idx==30:
                        print('Right occipital lobe')
                    if idx==31:
                        print('Right parietal lobe')
                    if idx==32:
                        print('Right putamen')
                    if idx==33:
                        print('Right temporal lobe')
                    if idx==34:
                        print('Right thalamus')
                    if idx==35:
                        print('Splenium of the corpus callosum')



                    if i=='Case 1':
                        print('Actual: Right parietal lobe\n')
                    if i=='Case 2':
                        print('Actual: Pons\n')
                    if i=='Case 3':
                        print('Actual: left cerebellar hemisphere\n')
                    if i=='Case 4':
                        print('Actual: posterior limb of the left internal capsule\n')
                    if i=='Case 5':
                        print('Actual: Left occipital lobe\n')
                    if i=='Case 6':
                        print('Actual: Left cerebral peduncle and left occipital lobe\n')
                    if i=='Case 7':
                        print('Actual: Right temporal lobe\n')        
                    if i=='Case 8':
                        print('Actual: Right frontal and parietal lobes, insula and external capsule\n')
                    if i=='Case 9':
                        print('Actual: Right corona radiata\n')
                    if i=='Case 10':
                        print('Actual: Left centrum semiovale\n')
                    if i=='Case 11':
                        print('Actual: Left thalamus\n')
                    if i=='Case 12':
                        print('Actual: Medulla on the right side\n')
                    if i=='Case 13':
                        print('Actual: Left temporal and parietal lobes\n')
                    if i=='Case 14':
                        print('Actual: Left corona radiata\n')
                    if i=='Case 15':
                        print('Actual: Right parietal lobe\n')
                    if i=='Case 16':
                        print('Actual: Left fronto-parietal lobe\n')
                    if i=='Case 17':
                        print('Actual: Left hippocampus\n')
                    if i=='Case 18':
                        print('Actual: Right parietal lobe\n')
                    if i=='Case 19':
                        print('Actual: Right parietal lobe\n')
                    if i=='Case 20':
                        print('Actual: Right cerebellar hemisphere\n')
                    if i=='Case 21':
                        print('Actual: Left cerebellar hemisphere\n')
                    if i=='Case 22':
                        print('Actual: Left thalamus\n')
                    if i=='Case 23':
                        print('Actual: Right centrum semiovale\n')
                    if i=='Case 24':
                        print('Actual: Right frontal lobe\n')
                    if i=='Case 25':
                        print('Actual: Left ganglio- capsular region\n')
                    if i=='Case 26':
                        print('Actual: Left frontal and parietal lobe\n')
                    if i=='Case 27':
                        print('Actual: Right precentral gyri\n')
                    if i=='Case 28':
                        print('Actual: Right frontal and parietal lobes\n')
                    if i=='Case 29':
                        print('Actual: Right fronto-parietal lobes, right caudate nucleus\n')
                    if i=='Case 30':
                        print('Actual: Right corona radiata\n')
                    if i=='Case 31':
                        print('Actual: Left lentiform nucleus\n')
                    if i=='Case 32':
                        print('Actual: Right thalamus\n')
                    if i=='Case 33':
                        print('Actual: Right parietal lobe\n')
                    if i=='Case 34':
                        print('Actual: Right cerebellar hemisphere\n')
                    if i=='Case 35':
                        print('Actual: Right thalamus\n')
                    if i=='Case 36':
                        print('Actual: Left centrum semiovale\n')
                
                

# Predicted class index.

if __name__ == '__main__':
    fileName=(sys.argv[2])
    fun(fileName)
"""
# decode the predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.applications.inception_v3 import decode_predictions
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
#scalar.fit(image)
print('Predicted:', decode_predictions(res, top=3)[0])
Xnew = scalar.transform(image)

#result = new_model.predict(Xnew)
#new_model.predict_classes(res.transpose())
"""
