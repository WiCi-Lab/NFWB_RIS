# Wideband Beamforming for RIS Assisted Near-Field Communications

We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers. Please refer to the following paper for more details.

@ARTICLE{10659326,
  author={Wang, Ji and Xiao, Jian and Zou, Yixuan and Xie, Wenwu and Liu, Yuanwei},,<br/>
  journal={IEEE Transactions on Wireless Communications}, ,<br/>
  title={Wideband Beamforming for RIS Assisted Near-Field Communications}, ,<br/>
  year={2024},,<br/>
  doi={10.1109/TWC.2024.3447570}}


### The deatailed description of each script is listed as follows.

* main_RIS_WB_SNR.py: the main function of the proposed E2E model.<br/>
* RIS_SUB_DIR_MIMO_NFWB_SNR.py: the SA-RIS architecture.<br/>
* RIS_TDD_DIR_MIMO_NFWB_SNR.py: the TTD-RIS architecture.<br/>
* RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py: the SA-RIS architecture with the quantified phase shift.<br/>
* RIS_TDD_DIR_MIMO_NFWB_SNR_R1.py: the TTD-RIS architecture with the quantified phase shift.<br/>
* PolarizedSelfAttention.py: polarized attention module.<br/>
* GFNet.py: learnable DFT module.<br/>
* Transformer_model.py: transformer module.<br/>
* NFBF_RIS_R3.py: channel generation functions.<br/>


### How to use this simulation code package?

1. You can run the “main_RIS_WB_SNR.py” script to obtain the desired results by switching the different beamforming models.
2. You can run the "NFBF_RIS_R3.py" script to generate a common test dataset at first so that provides a fairness performance comparision among various schemes.
3. When you call the "RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py" script for evaluating the beamforming performance under the case of discrete phase shift, you shoud pretrain a infinite beamforming model with the "RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py" script at first.
4. In the training stage, the different hyper-parameters setup will result in slight difference for final beamforming perfromance, e.g., the batchsize, the number of training epochs, and the training learning rate.
5. Now, this codes are a preliminary version composed of a few redundant statements, we will try my best to release the clean codes and add the necessary annotations in the future.


### Ackonwledge

We have integrated the model training and test code, and you can run the “main.py” file to obtain the channel estimation result of the LPAN or LPAN-L model. The detailed network model is given in the “LPAN.py” and “LPAN-L.py”.

Notes: 

(1)	Please confirm the required library files have been installed.

(2)	Please switch the desired data loading path and network models.

(3) In the training stage, the different hyper-parameters setup will result in slight difference for final channel estimation perfromance. According to our training experiences and some carried attempts, the hyper-parameters and network architecture can be further optimized to obtain better channel estimation performance gain, e.g., the dividing ratio between training samples and vadilation samples, the number of convolutional filters, the training learning rate, batchsize and epochs.

(4) Since the limitation of sample space (e.g., the fixed number of channel samples is collected for each user), the inevitable overfitting phenomenon may occur in the network training stage with the increase of epochs.

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.

