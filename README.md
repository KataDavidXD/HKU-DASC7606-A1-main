This project is a part of the HKU-DASC7606 course.
The train.py and test.py are modified for model improvement. To reproduce the result in the report, please use the following command:

CALL conda activate test_env_202401
CALL cd /d your path
CALL python train.py --coco_path ./data --output_path ./output --depth 101 --epochs 20 --b_size 4 --lr_plan 1 --lr_ini 1e-5 --lr_decay 0.1
CALL python test.py --coco_path ./data --checkpoint_path ./output/model_final.pt --depth 101 --set_name val 

Here lr_plan is to switch to the ReduceLROnPlateau scheduler.
Please contact yl9919@connect.hku.hk for any inquiries. 
