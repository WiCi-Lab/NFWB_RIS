# Wideband Beamforming for RIS Assisted Near-Field Communications

We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers. Please refer to the following paper for more details.

@ARTICLE{10659326,
  author={Wang, Ji and Xiao, Jian and Zou, Yixuan and Xie, Wenwu and Liu, Yuanwei},<br/>
  journal={IEEE Transactions on Wireless Communications} ,<br/>
  title={Wideband Beamforming for RIS Assisted Near-Field Communications},<br/>
  year={2024},<br/>
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

1. You can run the “main_RIS_WB_SNR.py” script to obtain the desired results by switching different beamforming models. <br/>
2. You can run the "NFBF_RIS_R3.py" script to generate a common test dataset at first so that provides a fairness performance comparison among various schemes. In this case, the system parameters between "NFBF_RIS_R3.py" and “main_RIS_WB_SNR.py” scripts must be consistent.<br/>
3. When you call the "RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py" script for evaluating the beamforming performance under the case of discrete phase shift, you should pretrain an infinite beamforming model with the "RIS_SUB_DIR_MIMO_NFWB_SNR_R1.py" script at first. <br/>
4. In the training stage, the different hyper-parameters will result in slight difference for final beamforming performance, e.g., the batchsize, the number of training epochs, and the training learning rate.<br/>
5. Now, this codes are a preliminary version composed of a few redundant statements, we will try my best to release the clean codes and add the necessary annotations in the future.<br/>


### Acknowledgements

We are very grateful for the following open-source repositories, which help us construct the beamforming model.<br/>
1.  https://github.com/DeLightCMU/PSA <br/>
2.  https://github.com/raoyongming/GFNet <br/>
3.  https://github.com/wuminghui123/DL_RSMA <br/>
