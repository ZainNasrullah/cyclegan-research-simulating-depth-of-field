# baseline
python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan --model cycle_gan --no_dropout

# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_loss --model cycle_gan --no_dropout --lambda_mask=1.0

# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_added --model cycle_gan --no_dropout --add_mask

