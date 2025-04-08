# test code
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
from torch.autograd import Variable
#from pytorch_wavelets import DWTForward,DWTInverse,DTCWTForward,DTCWTInverse
from net import network_fusion
import utils
from args import args
from loss_gradient import gradient_map,gradient_sobel64
import numpy as np
from contourlet import ContourDec, ContourRec,lpdec_layer,lprec_layer,dfbrec_layer
from ptflops import get_model_complexity_info
from thop import clever_format,profile
import time

def run_demo(model_fusion,ir_path,vi_path,output_path,name_ir,flag_img):
	img_ir, h, w, c = utils.get_test_image(ir_path, args.HEIGHT, args.WIDTH, flag=flag_img)
	img_vi, h, w, c = utils.get_test_image(vi_path, args.HEIGHT, args.WIDTH, flag=flag_img)

	if args.cuda:
		model_fusion.cuda()
		img_ir = img_ir.cuda()
		img_vi = img_vi.cuda()
	ct = ContourDec(nlevs=3).cuda()
	ict = ContourRec().cuda()
  
	gra_map64 = gradient_sobel64().cuda()
	img_vi_b=img_vi
	img_ir_b=img_ir
	
	s=img_vi_b.shape
	m=s[2]//16
	n=s[3]//16
	tic = time.time()
	vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h = model_fusion.vr.vi_encoder(img_vi_b[:,:,0:m*16,0:n*16],ct)#
	ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h = model_fusion.vr.ir_encoder(img_ir_b[:,:,0:m*16,0:n*16],ct)#0:m*16,0:n*16
  		
	for k in range(0,8):
		com_vi=F.pad(torch.abs(gra_map64(vi64_h[k])),(1,1,1,1),mode='replicate')#torch.ge(torch.abs(vi_h[k]),torch.abs(ir_h[k])).float()
		com_ir=F.pad(torch.abs(gra_map64(ir64_h[k])),(1,1,1,1),mode='replicate')#torch.ge(torch.abs(ir_h[k]),torch.abs(vi_h[k])).float()
		vi_mask=torch.ge(com_vi,com_ir).float()
		ir_mask=torch.ge(com_ir,com_vi).float()
		vi64_h[k]=vi_mask*vi64_h[k]+ir_mask*ir64_h[k]
	for k in range(0,8):
		com_vi=F.pad(torch.abs(gra_map64(vi32_h[k])),(1,1,1,1),mode='replicate')
		com_ir=F.pad(torch.abs(gra_map64(ir32_h[k])),(1,1,1,1),mode='replicate')
		vi_mask=torch.ge(com_vi,com_ir).float()
		ir_mask=torch.ge(com_ir,com_vi).float()
		vi32_h[k]=vi_mask*vi32_h[k]+ir_mask*ir32_h[k]		
	for k in range(0,8):
		com_vi=F.pad(torch.abs(gra_map64(vi16_h[k])),(1,1,1,1),mode='replicate')
		com_ir=F.pad(torch.abs(gra_map64(ir16_h[k])),(1,1,1,1),mode='replicate')
		vi_mask=torch.ge(com_vi,com_ir).float()
		ir_mask=torch.ge(com_ir,com_vi).float()
		vi16_h[k]=vi_mask*vi16_h[k]+ir_mask*ir16_h[k]

	f64_l,f32_l,f16_l,_,_,_,_,_,_,_,_,_ = model_fusion.feature_fusion(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l)

	out = model_fusion.vr.decoder(f64_l,f32_l,f16_l,vi64_h,vi32_h,vi16_h,ict)

	flops1,params1=profile(model_fusion.vr.vi_encoder,inputs=(img_vi_b[:,:,0:m*16,0:n*16],ct),verbose=False)
	flops2,params2=profile(model_fusion.feature_fusion,inputs=(vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l),verbose=False)
	flops3,params3=profile(model_fusion.vr.decoder,inputs=(f64_l,f32_l,f16_l,vi64_h,vi32_h,vi16_h, ict),verbose=False)
	flop=flops1*2+flops2+flops3
	param=params1*2+params2+params3
	flops,params=clever_format([flop,param],"%.3f")
	print(f"model flops:{flop}")
	print(f"model params:{param}")

	output_path_f = output_path + name_ir 
	utils.save_image_test(out[0,:,:,:], output_path_f)
	print(output_path_f)


def main():
	# False - gray True -RGB
	flag_img = False
	# ################# gray scale ########################################
	test_path = "./F2Fusion/M3FD/ir/"#测试图像的路径
	checkpoint_path = "./F2Fusion/checkpoint/Epoch_15_iters_1200.model"
	output_path = "./F2Fusion/fusion_results/Gray/" #融合结果的保存路径
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)
	with torch.no_grad():
				model_fusion = network_fusion()
				model_fusion.load_state_dict(torch.load(checkpoint_path,map_location='cuda'))
				model_fusion.eval()
				imgs_paths_ir, names = utils.list_images(test_path)
				num = len(imgs_paths_ir)
				for i in range(0,50):
					name_ir = names[i]
					ir_path = imgs_paths_ir[i]
					vi_path = ir_path.replace('ir', 'vi')					
					run_demo(model_fusion,ir_path,vi_path,output_path,name_ir,flag_img)                                               
				print(' visible and infrared Image Fusion Task Done......')

if __name__ == '__main__':
	main()


