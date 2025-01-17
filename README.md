The codes for "Machine-learning local resistive environments of dislocations in complex concentrated alloys from data generated by molecular dynamics simulations" is presented here.

To reproduce the paper, you can follow the steps.

1. Create an conda enviroment named hea-ana with python==3.8, and activate it.
```shell
conda create -n hea-ana python=3.8
conda activate hea-ana
```
2. Clone this repository, navigate to hea-md-ml-analysis folder and install all necessary packages.
```shell
git clone https://github.com/Ninazizi/hea-md-ml-analysis
cd hea-md-ml-analysis
pip install -r requirement.txt
```
3. Generate datapoints for PCC analysis and Machine Learning.
```shell
python load_data_edge_gradient_final.py
python load_data_screw_gradient_final.py
```
4. PCC analysis.
```shell
python plot_velocity.py
python plot_velocity-2.py
python plot_csfe.py
python plot_g.py
python plot_vel-csfe.py
python plot_vel-nye.py
```
5. Train the LightGBM model to predict the velocity.
```shell
python plot_vel-csfe_pred-crossvalidation.py
python abalation-prediction-acc.py
```   
