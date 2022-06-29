# X-Net

The code in this folder is adapted from the official X-Net repo [here](https://github.com/Andrewsher/X-Net)

### Running the trained model

1. To run the checkpoints, make sure your working directory is in this folder and create a data folder first:

    ```
    mkdir data
    ```

2. Download the `train.h5` file and `data_order.txt` file from [here]() and place them in the data folder

3. Prepare the checkpoint. Do:
    ```
    mkdir trained_model
    ```
    and download the trained checkpoints from [here]() and put them into the `trained_model` folder

4. Modify `bash/X-Net/predict.sh` if needed.

    `--data_file_path` : path to the training data

    `--pretrained_weight_file` : path to the pretrained weight checkpoint

    `--save_path` : path to save the predicted output

    `--data_order_path` : path to data_order.txt file

    and then run with:
    ```
    bash bash/X-Net/predict.sh
    ```

5. Modify `bash/X-Net/eval.sh` if needed.

    `--prediction_path` : path to the prediction produced by the model

    `--validatio_path` : path to the testing data

    `--log_path` : path to log the evaluation statistics.

    `--small_lesion_only` : whether to store visualization for small lesion samples only

    `--visual_path` : path to store the visualization results

    and then run with:
    ```
    bash bash/X-Net/eval.sh
    ```