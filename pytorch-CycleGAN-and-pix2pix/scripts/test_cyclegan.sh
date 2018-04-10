# baseline
#python test.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan --model cycle_gan --phase test --no_dropout

# baseline with mask and vgg loss
python test.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_loss_vgg --model cycle_gan --phase test --no_dropout --vgg19_loss --lambda_mask=1.0

# python test.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_added --model cycle_gan --phase test --no_dropout --add_mask
