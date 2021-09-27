# KiU-Net

This folder contains the adapted code from the official KiU-Net repo [here](https://github.com/jeya-maria-jose/KiU-Net-pytorch). 

### Running the trained checkpoints

1. To run the checkpoints, make sure your working directory is in this folder and create a data folder first:

    ```
    mkdir data
    ```

2. Download the `train.h5` file and `data_order.txt` file from [here]() and place them in the data folder

3. Then run:

    ```
    cd ../..
    bash bash/KiU-Net/process_data.sh
    ```

    The data file will be processed into train folder and validation folder. Do:
    ```
    ls external_src/KiU-Net/data
    ```
    and you should see 
    ```
    data_order.txt  train  train.h5  validation
    ```

4. Prepare the checkpoint. Do:
    ```
    cd external_src/KiU-Net
    mkdir trained_model
    ```
    and download the trained checkpoints from [here]() and put them into the `trained_model` folder

5. Modify the bash for prediction if needed. 

    `--loaddirec` : path to the trained checkpoint

    `--direc` : path to stored the predicted output

    `--visual_path` : path to store the visualization results

    `--small_lesion_only` : whether to produce visualizations for small lesion results only

    `--log_path` : path to store the evaluation statistics

6. Run:

    ```
    cd ../..
    bash bash/KiU-Net/predict.sh
    ```
    and the outputed segmentation will be in `external_src/KiU-Net/trained_model/outputs` in the format of a .npy files. The visualization will be in `external_src/KiU-Net/trained_model/visual`. For running speed the default setting is to only produce small lesion visualization. This setting can be changed in the bash script. The evaluation result is in `external_src/KiU-Net/trained_model/val_result.txt`