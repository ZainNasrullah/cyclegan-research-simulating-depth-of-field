python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_added --model cycle_gan --no_dropout --lambda_mask=2.0 --add_mask --continue_train

# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask --model cycle_gan --no_dropout --lambda_mask=1.0 --continue_train
# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan --model cycle_gan --no_dropout --continue_train
