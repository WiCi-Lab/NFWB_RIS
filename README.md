# Wideband Beamforming for RIS Assisted Near-Field Communications

We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers. Please refer to the following paper for more details.

@ARTICLE{10659326,
  author={Wang, Ji and Xiao, Jian and Zou, Yixuan and Xie, Wenwu and Liu, Yuanwei},,<br/>
  journal={IEEE Transactions on Wireless Communications}, ,<br/>
  title={Wideband Beamforming for RIS Assisted Near-Field Communications}, ,<br/>
  year={2024},,<br/>
  doi={10.1109/TWC.2024.3447570}}


### Script descriptions.

* main_RIS_WB_SNR.py: the main function of the proposed E2E model.<br/>
* RIS_SUB_DIR_MIMO_NFWB_SNR.py: the SA-RIS architecture.<br/>
* RIS_TDD_DIR_MIMO_NFWB_SNR.py: the TTD-RIS architecture.<br/>
* RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py: the SA-RIS architecture with the quantified phase shift.<br/>
* RIS_TDD_DIR_MIMO_NFWB_SNR_R1.py: the TTD-RIS architecture with the quantified phase shift.<br/>
* PolarizedSelfAttention.py: polarized attention module.<br/>
* GFNet.py: learnable DFT module.<br/>
* Transformer_model.py: transformer module.<br/>
* NFBF_RIS_R3.py: channel generation functions.<br/>


### Usage

1. You can run the “main_RIS_WB_SNR.py” script to obtain the desired results by switching the different beamforming models..<br/>
2. You can run the "NFBF_RIS_R3.py" script to generate a common test dataset at first so that provides a fairness performance comparision among various schemes..<br/>
3. When you call the "RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py" script for evaluating the beamforming performance under the case of discrete phase shift, you shoud pretrain a infinite beamforming model with the "RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py" script at first..<br/>
4. In the training stage, the different hyper-parameters setup will result in slight difference for final beamforming perfromance, e.g., the batchsize, the number of training epochs, and the training learning rate..<br/>
5. Now, this codes are a preliminary version composed of a few redundant statements, we will try my best to release the clean codes and add the necessary annotations in the future..<br/>


### Ackonwledge

We are very grateful for the following open-source repositories, which help us construct the proposed beamforming model.<br/>
1.  https://github.com/DeLightCMU/PSA, 
2.  https://github.com/raoyongming/GFNet
3.  https://github.com/wuminghui123/DL_RSMA

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.

