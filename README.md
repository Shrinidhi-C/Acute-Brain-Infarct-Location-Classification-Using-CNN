#ACUTE BRAIN INFARCTS CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS

"""A Deep-Learning project to detect Acute infarcts in different brain locations using Keras and Tensorflow libraries."""


The repository contains the source codes and the data folders as explained below:

Source codes :

     ● cropp.ipynb : The script for the data cleaning [To be executed first ]
  
     ● Acute_Brain_Infarcts_Classification_PartA.ipynb : The script for part A of the classification procedure.
  
     ● optimizer.py : The python script to convert the .h5 model to .pb and further on to IR model. It also freezes the model weights and graphs.
  
     ● infer.py : The minimal python script to obtain the inference results. It initializes the model and sets the parameters, target device and load the HW plugin. It reads the IR model and makes the executable environment.
  
Other folders:

     ● The CLEANED_DATA contains the images sorted according to the location of the acute infarct in the brain images. It has all 46 folders (all unique locations).

     ● The DWI_DATA is the folder to be used. All the images are cropped and resized to (128X128) pixels as well. This folder contains 36 folders.

     ● The Merged_Data folder contains the data after merging of the similar classes (manually) from DWI_DATA.

     ● Train_Data_31 : This folder has all images being augmented with 30 copies per class .This has to be used as the training data (X_train) , with the
respective folder's name (location itself) as a label for the same (y_train).

     ● test : This folder has 10 folders (only repeated locations on merging the similar classes) . This has to be used as the test data with the respective folder's name (location itself) as a label for the same( y_test).


PROCEDURE:

● Run the cropp.ipynb which creates two folders:

    i) cropped_DATA: this folder contains all the images as in CLEANED_DATA which are cropped and stored in the same format
  
    ii) DWI_DATA: this folder contains only the DWI images from the cropped_DATA
  
● Run the Acute_Brain_Infarcts_Classification_PartA.ipynb

    * Setup the environment using the command:

    cd C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\bin
    setupvars.bat

    * Further convert it to the IR model using the following commands.

● python optimizer.py

● python "C:\Program Files(x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo_tf.py" --input_model
"C:\Users\Dell\Envs\PES_AI_PIP\model\pro_dl_frozen_model.pb"--output_dir './model' --input_shape "[1,128,128,3]" --data_type FP32

* The optimizer takes the path to mo_tf.py, the .pb file of the model, shape of the input fed and the data type as the arguments.
Next the model is to be executed and the predictions to be made on the inference data using:

● python infer.py -i "C:\Users\Dell\Envs\PES_AI_PIP\Inference_Data" -d "CPU" -m
"C:\Users\Dell\Envs\PES_AI_PIP\model\pro_dl_frozen_model.xml"

* The infer.py takes the path of the inference data, target device and the .xml file of the model as the arguments.
(note: All the commands to be executed in command prompt in administrative mode under the virtual environment)


* Output_weights : The file has the final output weights that are frozen and the model is used for prediction.

* Names.txt : The file has the names and usn of the team members

* model : Contains the .xml, ,pb, .bin, .mapping files.

* Inference_Data : The unseen images to be predicted by the model.

* Inference_results : The output of the Inference [along with commands
