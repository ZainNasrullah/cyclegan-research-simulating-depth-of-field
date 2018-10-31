# # baseline
# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan --model cycle_gan --no_dropout

# # baseline with mask and vgg loss
# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_loss_vgg --model cycle_gan --no_dropout --vgg19_loss --lambda_mask=1.0

# # baseline with add_mask enabled
# python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_added --model cycle_gan --no_dropout --add_mask

# baseline with mask loss and add_mask enabled
python train.py --dataroot ./datasets/selfie2bokeh --name selfie2bokeh_cyclegan_mask_added_mask_loss_vgg --model cycle_gan --no_dropout --add_mask --vgg19_loss --lambda_mask=0.5

