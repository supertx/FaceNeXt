# FaceNeXt
combine mobilefacenet and ConNeXt, we proudly release FaceNeXt, which may reach a higher performance on many stages.

🤗update: 2025-07-26 pretrain part finish. Which is a self-supervised learning based training
* pretrained visualization
![pretrain visualization](./data/img.png)

🤗update: 2024-07-30 finetune part finish. use arcface loss to train the pretrain self-supervised learning model
* beautiful loss curve😀 

![loss curve](./data/img_1.png)

🤗update: 2024-8-5 fix arcface loss bug, add cosface loss together

😊update:2024-8-16 add arcface loss and gan loss at pretrain phrase, made some experiments on the performance of the model
* 25 epoch finetune on MS1MV3(with only 8mb model)

![accuracy](./data/img_2.png)

✂️update:2024-8-20 reorganize face dataset MS1MV3, have the face landmarks in the train rec
pretrain phrase use the face landmarks to do the data mask, on use face loss that seams reach a better result
finetune frozen model parameters in the first x epoch.
the project may have no upgrade in the future
if u are interested in this project, feel free to contact me on 1374411672@qq.com